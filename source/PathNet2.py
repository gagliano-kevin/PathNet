"""
PathNet.py
==========
Core logic for training quantized MLPs via A*-based heuristic search.

Public Classes
--------------
QuantizedMLP
    Wraps a ``torch.nn.Module`` with optional fixed-point quantisation and
    provides evaluation, weight-hashing, and state-management utilities.

SearchNode
    Immutable value-object representing a single node in the A* weight-space
    search graph.

Trainer
    A*-search trainer that uses a single global sliding-window kernel for
    neighbourhood generation.

TrainerLayerWiseKernel
    A*-search trainer that uses independent per-layer kernels and strides.

TrainerRandomSampling
    A*-search trainer that uses stochastic parameter perturbation for
    neighbourhood generation (no kernel or stride required).
"""

from __future__ import annotations

import heapq
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import Tensor
from torch.nn import Module

from source.utils.memory_guard import SystemMemoryGuard
from source.utils.neighbors_utils import (
    get_neighbors,
    get_neighbors_layer_wise,
    get_neighbors_random,
)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

StateHash = tuple[int, ...]
NeighborList = list[tuple["QuantizedMLP", float]]


# ---------------------------------------------------------------------------
# QuantizedMLP
# ---------------------------------------------------------------------------


class QuantizedMLP:
    """A PyTorch model wrapper that rounds parameters to a fixed-point grid.

    Parameters
    ----------
    model:
        The underlying ``torch.nn.Module`` whose parameters will be quantised.
    loss_fn:
        Callable ``(predictions, targets) -> scalar Tensor`` used for
        evaluation.
    quantization_factor:
        Reciprocal of the quantisation step size.  E.g. ``10`` means weights
        are rounded to the nearest multiple of ``0.1``.
    parameter_range:
        ``(min, max)`` hard clip range.  Values outside this range are clamped
        before quantisation; if encountered during single-tensor quantisation
        the model is invalidated and ``overflow`` is set.
    enable_quantization:
        When ``False`` the model is stored as-is without any rounding applied.
    debug:
        When ``True`` overflow and out-of-range events are reported to stdout.
    """

    __slots__ = (
        "model",
        "loss_fn",
        "quantization_factor",
        "parameter_range",
        "overflow",
        "enable_quantization",
        "debug",
    )

    def __init__(
        self,
        model: Module,
        loss_fn,
        quantization_factor: float = 10.0,
        parameter_range: tuple[float, float] = (-10.0, 10.0),
        enable_quantization: bool = True,
        debug: bool = False,
    ) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.quantization_factor = quantization_factor
        self.parameter_range = parameter_range
        self.overflow: bool = False
        self.enable_quantization = enable_quantization
        self.debug = debug

        if self.enable_quantization:
            self.quantize()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def quantize(self) -> None:
        """Quantise all model parameters in-place.

        Each parameter tensor is first clipped to ``parameter_range``, then
        rounded to the nearest multiple of ``1 / quantization_factor``.
        """
        lo, hi = self.parameter_range
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if torch.any(param.data < lo) or torch.any(param.data > hi):
                    if self.debug:
                        print(
                            f"[QuantizedMLP] Parameter '{name}' is outside "
                            f"{self.parameter_range} — clipping."
                        )
                    param.data.clamp_(lo, hi)
                param.data.mul_(self.quantization_factor).round_().div_(
                    self.quantization_factor
                )

    def quantize_tensor(self, tensor_idx: int) -> None:
        """Quantise a single parameter tensor in-place.

        If the tensor contains values outside ``parameter_range`` the model is
        invalidated: ``self.model`` is set to ``None`` and ``self.overflow`` is
        raised to ``True``.

        Parameters
        ----------
        tensor_idx:
            Zero-based index into ``list(self.model.parameters())``.
        """
        lo, hi = self.parameter_range
        with torch.no_grad():
            tensor = list(self.model.parameters())[tensor_idx]
            if torch.any(tensor.data < lo) or torch.any(tensor.data > hi):
                if self.debug:
                    print(
                        f"[QuantizedMLP] Parameter at index {tensor_idx} "
                        f"is out of range — setting overflow flag."
                    )
                self.overflow = True
                self.model = None
                return
            tensor.data.mul_(self.quantization_factor).round_().div_(
                self.quantization_factor
            )

    def evaluate(self, X: Tensor, Y: Tensor) -> float:
        """Compute the scalar loss on a ``(X, Y)`` batch.

        Returns
        -------
        float
            The scalar loss value.

        Raises
        ------
        ValueError
            If the model has been invalidated by a prior overflow event.
        """
        self._assert_valid()
        self.model.eval()
        with torch.no_grad():
            return self.loss_fn(self.model(X), Y).item()

    def get_flat_weights(self) -> Tensor:
        """Return a 1-D tensor of all model parameters concatenated.

        Raises
        ------
        ValueError
            If the model has been invalidated by a prior overflow event.
        """
        self._assert_valid()
        return torch.cat([p.detach().flatten() for p in self.model.parameters()])

    def get_state_hash(self) -> StateHash:
        """Return a hashable tuple that uniquely identifies the current weight state.

        Weights are scaled by ``quantization_factor`` and cast to ``long`` so
        that equality comparisons are integer-exact.

        Raises
        ------
        ValueError
            If the model has been invalidated by a prior overflow event.
        """
        self._assert_valid()
        return tuple(
            (self.get_flat_weights() * self.quantization_factor).long().tolist()
        )

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __str__(self) -> str:
        lines = [
            (
                f"QuantizedMLP("
                f"quantization_factor={self.quantization_factor}, "
                f"parameter_range={self.parameter_range}, "
                f"overflow={self.overflow})"
            ),
            "Model Parameters:",
        ]
        for name, param in self.model.named_parameters():
            lines.append(f"  {name}: {param.data}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"QuantizedMLP("
            f"quantization_factor={self.quantization_factor!r}, "
            f"parameter_range={self.parameter_range!r}, "
            f"overflow={self.overflow!r})"
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _assert_valid(self) -> None:
        """Raise ``ValueError`` if this instance is in an overflow state."""
        if self.model is None or self.overflow:
            raise ValueError(
                "QuantizedMLP is invalid: the model was set to None after an "
                "overflow event during quantization."
            )


# ---------------------------------------------------------------------------
# SearchNode
# ---------------------------------------------------------------------------


@dataclass
class SearchNode:
    """A node in the A* weight-space search graph.

    Attributes
    ----------
    quantized_mlp:
        The ``QuantizedMLP`` state represented by this node.
    g_val:
        Accumulated cost-to-come from the initial (root) node.
    h_val:
        Heuristic cost-to-go, taken as the model's current evaluation loss.
    parent:
        Reference to the predecessor node, or ``None`` for the root.
    f_val:
        Total estimated cost ``g_val + h_val``; derived automatically in
        ``__post_init__``.
    """

    quantized_mlp: QuantizedMLP
    g_val: float
    h_val: float
    parent: Optional[SearchNode] = field(default=None, compare=False, repr=False)
    f_val: float = field(init=False)

    def __post_init__(self) -> None:
        self.f_val = self.g_val + self.h_val

    def __lt__(self, other: SearchNode) -> bool:
        """Enable min-heap ordering by total estimated cost."""
        return self.f_val < other.f_val


# ---------------------------------------------------------------------------
# _BaseTrainer  —  internal abstract base class
# ---------------------------------------------------------------------------


class _BaseTrainer(ABC):
    """Abstract base class shared by all A*-search MLP trainers.

    Subclasses must implement :py:meth:`_get_neighbors` to define their
    neighbourhood-generation strategy.  Subclasses that support dynamic
    kernel reshaping should also override :py:meth:`_apply_kernel_reshape`.

    The complete A* iteration logic — including beam-search pruning, early
    stopping, dynamic quantisation, and timing — lives here and is reused
    by every concrete trainer without duplication.
    """

    def __init__(
        self,
        model: Module,
        loss_fn,
        quantization_factor: float,
        parameter_range: tuple[float, float],
        debug_mlp: bool = True,
        # Early stopping
        early_stopping: bool = False,
        e_s_patience: int = 250,
        # Dynamic quantisation
        dynamic_quantization: bool = False,
        d_q_patience: int = 100,
        quantization_factor_multiplier: float = 10.0,
        max_quantization_factor: float = 1e4,
        # Dynamic kernel reshaping
        dynamic_kernel_reshaping: bool = False,
        d_k_r_patience: int = 100,
        # Thresholds
        loss_improvement_threshold: float = 1e-3,
        # General
        max_iterations: int = 1000,
        log_freq: int = 1000,
        measure_time: bool = True,
        save_trained_model: bool = False,
        model_name: str = "best_model",
    ) -> None:
        self.memory_guard = SystemMemoryGuard()

        # --- Core ---
        self.model = model
        self.loss_fn = loss_fn
        self.quantization_factor = quantization_factor
        self.parameter_range = parameter_range
        self.debug_mlp = debug_mlp

        # --- Early stopping ---
        self.early_stopping = early_stopping
        self.e_s_patience = e_s_patience
        self.e_s_wait: int = 0

        # --- Dynamic quantisation ---
        self.dynamic_quantization = dynamic_quantization
        self.d_q_patience = d_q_patience
        self.quantization_factor_multiplier = quantization_factor_multiplier
        self.max_quantization_factor = max_quantization_factor
        self.d_q_wait: int = 0

        # --- Dynamic kernel reshaping (counter lives here; state in subclasses) ---
        self.dynamic_kernel_reshaping = dynamic_kernel_reshaping
        self.d_k_r_patience = d_k_r_patience
        self.d_k_r_wait: int = 0

        # --- Convergence & iteration control ---
        self.loss_improvement_threshold = loss_improvement_threshold
        self.max_iterations = max_iterations
        self.log_freq = log_freq

        # --- Persistence ---
        self.measure_time = measure_time
        self.save_trained_model = save_trained_model
        self.model_name = model_name

        # --- Search state (mutated during training) ---
        self.open_set: list[tuple[float, SearchNode]] = []
        self.g_costs: dict[StateHash, float] = {}
        self.best_node: Optional[SearchNode] = None

        # --- Training metrics ---
        self.loss_history: list[float] = []
        self.f_history: list[float] = []
        self.g_history: list[float] = []
        self.training_time: Optional[float] = None

        self.dynamic_adjustments_log: dict[str, list[int]] = {
            "dynamic_quantization_iterations": [],
            "dynamic_kernel_reshaping_iterations": [],
        }

    # ------------------------------------------------------------------
    # Abstract interface — must be implemented by each concrete subclass
    # ------------------------------------------------------------------

    @abstractmethod
    def _get_neighbors(
        self, node: SearchNode, X: Tensor, Y: Tensor
    ) -> NeighborList:
        """Return a list of ``(QuantizedMLP, loss)`` neighbours for *node*.

        Parameters
        ----------
        node:
            The node being expanded.
        X, Y:
            Training batch used to evaluate each candidate neighbour.
        """

    def _apply_kernel_reshape(self) -> bool:
        """Attempt to shrink kernels / strides; return ``True`` if anything changed.

        The default implementation is a no-op (returns ``False``).  Subclasses
        that support dynamic kernel reshaping should override this method.
        """
        return False

    # ------------------------------------------------------------------
    # Public training entry-points
    # ------------------------------------------------------------------

    def train(self, X: Tensor, Y: Tensor) -> None:
        """Train via the standard A* search loop (no open-set size bound).

        Parameters
        ----------
        X:
            Input training tensor.
        Y:
            Target training tensor.
        """
        self._run_search_loop(X, Y, beam_width=None)

    def beam_search_opt_train(
        self, X: Tensor, Y: Tensor, beam_width: int = 500
    ) -> None:
        """Train via beam-search-optimised A* to bound peak memory usage.

        The open set is pruned to *beam_width* entries after every expansion,
        and ``g_costs`` is periodically synchronised to prevent unbounded
        dictionary growth.

        Parameters
        ----------
        X:
            Input training tensor.
        Y:
            Target training tensor.
        beam_width:
            Maximum number of nodes retained in the open set.
        """
        self._run_search_loop(X, Y, beam_width=int(beam_width))

    # ------------------------------------------------------------------
    # Model persistence
    # ------------------------------------------------------------------

    def save_model(self, filename: str = "best_model.pth") -> None:
        """Persist the best model's ``state_dict`` to *filename*.

        Parameters
        ----------
        filename:
            Destination file path.
        """
        if self.best_node is not None:
            torch.save(
                self.best_node.quantized_mlp.model.state_dict(), filename
            )
            print(f"Best model saved to '{filename}'.")
        else:
            print("No best model to save.")

    def load_model(
        self,
        model_architecture: Module,
        loss_fn,
        quantization_factor: float = 10.0,
        parameter_range: tuple[float, float] = (-5.0, 5.0),
        enable_quantization: bool = True,
        debug: bool = False,
        filename: str = "best_model.pth",
    ) -> QuantizedMLP:
        """Load a ``state_dict`` from *filename* and return a :class:`QuantizedMLP`.

        Parameters
        ----------
        model_architecture:
            An uninitialised model whose architecture matches the saved weights.
        loss_fn:
            Loss function to attach to the returned wrapper.
        quantization_factor:
            Quantisation factor for the loaded model.
        parameter_range:
            Clip range for the loaded model.
        enable_quantization:
            Whether to apply quantisation after loading.
        debug:
            Enable overflow debug output on the loaded wrapper.
        filename:
            Source file path.

        Returns
        -------
        QuantizedMLP
            A fully initialised wrapper around *model_architecture*.
        """
        state_dict = torch.load(filename, weights_only=True)
        model_architecture.load_state_dict(state_dict)
        quantized_mlp = QuantizedMLP(
            model_architecture,
            loss_fn,
            quantization_factor,
            parameter_range,
            enable_quantization,
            debug,
        )
        print(f"Model loaded from '{filename}'.")
        return quantized_mlp

    # ------------------------------------------------------------------
    # Dynamic-adaptation control
    # ------------------------------------------------------------------

    def reset_dynamic_counters(self) -> None:
        """Reset all patience counters after a meaningful loss improvement."""
        if self.early_stopping:
            self.e_s_wait = 0
        if self.dynamic_quantization:
            self.d_q_wait = 0
        if self.dynamic_kernel_reshaping:
            self.d_k_r_wait = 0

    def increment_dynamic_counters(self, iteration: int) -> None:
        """Advance patience counters and trigger any pending adaptations.

        Called once per non-improving iteration.  Manages dynamic quantisation
        and dynamic kernel reshaping according to their respective patience
        thresholds.

        Parameters
        ----------
        iteration:
            Current 0-indexed iteration number, recorded in the adjustments log.
        """
        if self.early_stopping:
            self.e_s_wait += 1

        if self.dynamic_quantization:
            self.d_q_wait += 1
            if self.d_q_wait >= self.d_q_patience:
                self._trigger_quantization_increase(iteration)

        if self.dynamic_kernel_reshaping:
            self.d_k_r_wait += 1
            if self.d_k_r_wait >= self.d_k_r_patience:
                self._trigger_kernel_reshape(iteration)

    # ------------------------------------------------------------------
    # Core search loop  (shared by train and beam_search_opt_train)
    # ------------------------------------------------------------------

    def _run_search_loop(
        self, X: Tensor, Y: Tensor, beam_width: Optional[int]
    ) -> None:
        """Execute the A* iteration loop.

        This single implementation is shared by both :meth:`train` (pure A*,
        ``beam_width=None``) and :meth:`beam_search_opt_train` (bounded open
        set, ``beam_width=<int>``).

        Parameters
        ----------
        X, Y:
            Training batch.
        beam_width:
            If ``None`` no pruning is applied.  Otherwise the open set is
            trimmed to this many entries after every expansion step.
        """
        start_time = time.perf_counter() if self.measure_time else 0.0
        self._seed_open_set(X, Y)

        for iteration in range(self.max_iterations):
            if not self.open_set:
                print("Open set is empty. Terminating search.")
                break

            _f, current_node = heapq.heappop(self.open_set)
            current_hash = current_node.quantized_mlp.get_state_hash()

            # Discard stale entries whose path has since been superseded.
            # Note: the first condition should always be False in practice; it
            # is retained as a defensive guard against key-lookup errors.
            if (
                current_hash not in self.g_costs
                or current_node.g_val > self.g_costs[current_hash]
            ):
                continue

            self.loss_history.append(current_node.h_val)
            self.f_history.append(current_node.f_val)
            self.g_history.append(current_node.g_val)

            # --- Dynamic-adaptation gate ---
            improvement = self.best_node.h_val - current_node.h_val
            if improvement > self.loss_improvement_threshold:
                self.reset_dynamic_counters()
            else:
                self.increment_dynamic_counters(iteration)

            if self.early_stopping and self.e_s_wait >= self.e_s_patience:
                print(
                    f"Early stopping triggered after {self.e_s_patience} "
                    "iterations without improvement."
                )
                break

            if self.memory_guard.memory_exceeded():
                print(
                    "Memory usage exceeded threshold. "
                    "Terminating training to prevent system instability."
                )
                break

            # --- Track global best ---
            if current_node.h_val < self.best_node.h_val:
                self.best_node = current_node
                print(
                    f"Iteration {iteration + 1}: "
                    f"New best loss = {self.best_node.h_val}"
                )

            if (iteration + 1) % self.log_freq == 0:
                print(
                    f"Iteration {iteration + 1}: "
                    f"Best current loss = {self.best_node.h_val}"
                )

            # --- Expand neighbours ---
            for neighbor_mlp, neighbor_loss in self._get_neighbors(
                current_node, X, Y
            ):
                if neighbor_mlp.overflow:
                    continue

                neighbor_hash = neighbor_mlp.get_state_hash()
                # Edge cost: c(n, n') = loss(n') - loss(n)
                g = current_node.g_val + (neighbor_loss - current_node.h_val)

                if (
                    neighbor_hash not in self.g_costs
                    or g < self.g_costs[neighbor_hash]
                ):
                    self.g_costs[neighbor_hash] = g
                    new_node = SearchNode(
                        neighbor_mlp,
                        g_val=g,
                        h_val=neighbor_loss,
                        parent=current_node,
                    )
                    heapq.heappush(self.open_set, (new_node.f_val, new_node))

            # --- Beam-search pruning (beam mode only) ---
            if beam_width is not None:
                if len(self.open_set) > beam_width:
                    self.open_set = heapq.nsmallest(
                        beam_width, self.open_set, key=lambda x: x[0]
                    )
                    heapq.heapify(self.open_set)

                # Periodically synchronise g_costs with the bounded open set
                # to prevent unbounded dictionary growth.
                if iteration % 50 == 0:
                    active_hashes: set[StateHash] = {
                        node.quantized_mlp.get_state_hash()
                        for _, node in self.open_set
                    }
                    # Always retain the best node's hash as an anchor.
                    active_hashes.add(
                        self.best_node.quantized_mlp.get_state_hash()
                    )
                    self.g_costs = {
                        h: self.g_costs[h]
                        for h in active_hashes
                        if h in self.g_costs
                    }

        print(f"Search completed after {iteration + 1} iterations.")
        print(f"Best loss found: {self.best_node.h_val}")

        if self.measure_time:
            self.training_time = time.perf_counter() - start_time
            print(f"Total training time: {self.training_time:.4f} seconds")

        if self.save_trained_model:
            self.save_model(filename=f"{self.model_name}.pth")

    def _seed_open_set(self, X: Tensor, Y: Tensor) -> None:
        """Create the root ``SearchNode`` and push it onto the open set."""
        initial_mlp = QuantizedMLP(
            self.model,
            self.loss_fn,
            self.quantization_factor,
            self.parameter_range,
            debug=self.debug_mlp,
        )
        initial_loss = initial_mlp.evaluate(X, Y)
        initial_node = SearchNode(
            quantized_mlp=initial_mlp, g_val=0.0, h_val=initial_loss
        )
        initial_hash = initial_mlp.get_state_hash()

        heapq.heappush(self.open_set, (initial_node.f_val, initial_node))
        self.g_costs[initial_hash] = initial_node.g_val
        self.best_node = initial_node

    # ------------------------------------------------------------------
    # Private dynamic-adaptation helpers
    # ------------------------------------------------------------------

    def _trigger_quantization_increase(self, iteration: int) -> None:
        """Increase the quantisation factor if still below the maximum."""
        new_qf = min(
            self.quantization_factor * self.quantization_factor_multiplier,
            self.max_quantization_factor,
        )
        if new_qf > self.quantization_factor:
            self.dynamic_adjustments_log[
                "dynamic_quantization_iterations"
            ].append(iteration)
            print(
                f"Dynamic Quantization: {self.quantization_factor} -> {new_qf}"
            )
            self.quantization_factor = new_qf
        self.d_q_wait = 0

    def _trigger_kernel_reshape(self, iteration: int) -> None:
        """Delegate kernel shrinkage to the subclass and log if it occurred."""
        modification_occurred = self._apply_kernel_reshape()
        self.d_k_r_wait = 0
        if modification_occurred:
            self.dynamic_adjustments_log[
                "dynamic_kernel_reshaping_iterations"
            ].append(iteration)
            print("Dynamic Kernel Reshaping applied to kernels and strides.")


# ---------------------------------------------------------------------------
# Trainer  —  global sliding-window kernel
# ---------------------------------------------------------------------------


class Trainer(_BaseTrainer):
    """A*-search trainer using a single global sliding-window kernel.

    Neighbourhood generation is delegated to :func:`get_neighbors`.

    Parameters
    ----------
    model:
        PyTorch model to train.
    loss_fn:
        Loss callable ``(predictions, targets) -> scalar Tensor``.
    quantization_factor:
        Reciprocal step size for weight quantisation.
    parameter_range:
        ``(min, max)`` hard clip range for all weights.
    debug_mlp:
        Forward debug flag to each :class:`QuantizedMLP` instance created
        during search.
    weight_kernel:
        ``[rows, cols]`` sliding-window kernel applied to weight matrices.
        Defaults to ``[2, 2]``.
    bias_kernel:
        ``[size]`` sliding-window kernel applied to bias vectors.
        Defaults to ``[2]``.
    x_stride:
        Horizontal step size for the kernel scan.
    y_stride:
        Vertical step size for the kernel scan.
    delta_abs:
        Optional absolute perturbation magnitude override.
    early_stopping:
        Enable patience-based early stopping.
    e_s_patience:
        Non-improving iterations tolerated before stopping.
    dynamic_quantization:
        Automatically increase quantisation resolution on plateau.
    d_q_patience:
        Plateau iterations before increasing quantisation resolution.
    quantization_factor_multiplier:
        Multiplicative factor applied to ``quantization_factor`` on plateau.
    max_quantization_factor:
        Upper bound for the dynamically adjusted quantisation factor.
    dynamic_kernel_reshaping:
        Automatically shrink kernels and strides on plateau.
    d_k_r_patience:
        Plateau iterations before shrinking kernels.
    x_weight_kernel_decr:
        Column-dimension decrement applied to ``weight_kernel`` per reshape.
    y_weight_kernel_decr:
        Row-dimension decrement applied to ``weight_kernel`` per reshape.
    y_bias_kernel_decr:
        Decrement applied to ``bias_kernel`` per reshape.
    min_weight_kernel:
        Lower bounds ``[min_rows, min_cols]`` for ``weight_kernel``.
    min_bias_kernel:
        Lower bound ``[min_size]`` for ``bias_kernel``.
    x_stride_decr:
        Decrement applied to ``x_stride`` per reshape.
    y_stride_decr:
        Decrement applied to ``y_stride`` per reshape.
    min_x_stride:
        Lower bound for ``x_stride``.
    min_y_stride:
        Lower bound for ``y_stride``.
    loss_improvement_threshold:
        Minimum loss delta counted as a genuine improvement.
    max_iterations:
        Hard cap on the number of search iterations.
    log_freq:
        Progress logging frequency (every *n* iterations).
    measure_time:
        Record and print wall-clock training duration.
    save_trained_model:
        Automatically save the best model after training completes.
    model_name:
        Base filename (without ``.pth`` extension) used by :meth:`save_model`.
    """

    def __init__(
        self,
        model: Module,
        loss_fn,
        quantization_factor: float,
        parameter_range: tuple[float, float],
        debug_mlp: bool = True,
        # Neighbourhood generation
        weight_kernel: Optional[list[int]] = None,
        bias_kernel: Optional[list[int]] = None,
        x_stride: int = 1,
        y_stride: int = 1,
        delta_abs: Optional[float] = None,
        # Early stopping
        early_stopping: bool = False,
        e_s_patience: int = 250,
        # Dynamic quantisation
        dynamic_quantization: bool = False,
        d_q_patience: int = 100,
        quantization_factor_multiplier: float = 10.0,
        max_quantization_factor: float = 1e4,
        # Dynamic kernel reshaping
        dynamic_kernel_reshaping: bool = False,
        d_k_r_patience: int = 100,
        x_weight_kernel_decr: int = 1,
        y_weight_kernel_decr: int = 1,
        y_bias_kernel_decr: int = 1,
        min_weight_kernel: Optional[list[int]] = None,
        min_bias_kernel: Optional[list[int]] = None,
        x_stride_decr: int = 1,
        y_stride_decr: int = 1,
        min_x_stride: int = 1,
        min_y_stride: int = 1,
        # Thresholds
        loss_improvement_threshold: float = 1e-3,
        # General
        max_iterations: int = 1000,
        log_freq: int = 1000,
        measure_time: bool = True,
        save_trained_model: bool = False,
        model_name: str = "best_model",
    ) -> None:
        super().__init__(
            model=model,
            loss_fn=loss_fn,
            quantization_factor=quantization_factor,
            parameter_range=parameter_range,
            debug_mlp=debug_mlp,
            early_stopping=early_stopping,
            e_s_patience=e_s_patience,
            dynamic_quantization=dynamic_quantization,
            d_q_patience=d_q_patience,
            quantization_factor_multiplier=quantization_factor_multiplier,
            max_quantization_factor=max_quantization_factor,
            dynamic_kernel_reshaping=dynamic_kernel_reshaping,
            d_k_r_patience=d_k_r_patience,
            loss_improvement_threshold=loss_improvement_threshold,
            max_iterations=max_iterations,
            log_freq=log_freq,
            measure_time=measure_time,
            save_trained_model=save_trained_model,
            model_name=model_name,
        )

        # Defensive copies prevent the caller's lists from being mutated during
        # dynamic kernel reshaping.
        self.weight_kernel: list[int] = (
            list(weight_kernel) if weight_kernel is not None else [2, 2]
        )
        self.bias_kernel: list[int] = (
            list(bias_kernel) if bias_kernel is not None else [2]
        )
        self.x_stride = x_stride
        self.y_stride = y_stride
        self.delta_abs = delta_abs

        # Kernel-reshaping configuration
        self.x_weight_kernel_decr = x_weight_kernel_decr
        self.y_weight_kernel_decr = y_weight_kernel_decr
        self.y_bias_kernel_decr = y_bias_kernel_decr
        self.min_weight_kernel: list[int] = (
            list(min_weight_kernel) if min_weight_kernel is not None else [1, 1]
        )
        self.min_bias_kernel: list[int] = (
            list(min_bias_kernel) if min_bias_kernel is not None else [1]
        )
        self.x_stride_decr = x_stride_decr
        self.y_stride_decr = y_stride_decr
        self.min_x_stride = min_x_stride
        self.min_y_stride = min_y_stride

    # ------------------------------------------------------------------
    # Abstract-method implementations
    # ------------------------------------------------------------------

    def _get_neighbors(
        self, node: SearchNode, X: Tensor, Y: Tensor
    ) -> NeighborList:
        return get_neighbors(
            node,
            X,
            Y,
            self.quantization_factor,
            self.weight_kernel,
            self.bias_kernel,
            self.x_stride,
            self.y_stride,
            self.delta_abs,
        )

    def _apply_kernel_reshape(self) -> bool:
        """Shrink the global weight / bias kernels and strides toward their minima."""
        modified = False

        if self.weight_kernel[0] > self.min_weight_kernel[0]:
            self.weight_kernel[0] = max(
                self.weight_kernel[0] - self.y_weight_kernel_decr,
                self.min_weight_kernel[0],
            )
            modified = True
        if self.weight_kernel[1] > self.min_weight_kernel[1]:
            self.weight_kernel[1] = max(
                self.weight_kernel[1] - self.x_weight_kernel_decr,
                self.min_weight_kernel[1],
            )
            modified = True

        if self.bias_kernel[0] > self.min_bias_kernel[0]:
            self.bias_kernel[0] = max(
                self.bias_kernel[0] - self.y_bias_kernel_decr,
                self.min_bias_kernel[0],
            )
            modified = True

        if self.x_stride > self.min_x_stride:
            self.x_stride = max(
                self.x_stride - self.x_stride_decr, self.min_x_stride
            )
            modified = True
        if self.y_stride > self.min_y_stride:
            self.y_stride = max(
                self.y_stride - self.y_stride_decr, self.min_y_stride
            )
            modified = True

        return modified

    def dynamic_reshape_kernels_and_strides(self) -> bool:
        """Shrink kernels and strides toward their configured minima.

        .. deprecated::
            Call :meth:`_apply_kernel_reshape` directly in new code.  This
            public alias is retained for backward compatibility only.
        """
        return self._apply_kernel_reshape()


# ---------------------------------------------------------------------------
# TrainerLayerWiseKernel  —  per-layer kernels and strides
# ---------------------------------------------------------------------------


class TrainerLayerWiseKernel(_BaseTrainer):
    """A*-search trainer with independent per-layer kernels and strides.

    Uses :func:`get_neighbors_layer_wise` for neighbourhood generation,
    enabling fine-grained control over how each weight matrix and bias vector
    is perturbed.

    Parameters that differ from :class:`Trainer`
    ---------------------------------------------
    weight_kernels:
        List of ``[rows, cols]`` kernels — one entry per weight layer.
        Defaults to ``[[2, 2]]``.
    bias_kernels:
        List of ``[size]`` kernels — one entry per bias layer.
        Defaults to ``[[2]]``.
    weight_strides:
        List of ``[x_stride, y_stride]`` strides — one entry per weight layer.
        Defaults to ``[[1, 1]]``.
    bias_strides:
        List of ``[stride]`` strides — one entry per bias layer.
        Defaults to ``[[1]]``.
    min_weight_kernel, min_bias_kernel:
        Global minimum kernel sizes applied uniformly across all layers during
        dynamic reshaping.

    All other parameters are identical to :class:`Trainer`.
    """

    def __init__(
        self,
        model: Module,
        loss_fn,
        quantization_factor: float,
        parameter_range: tuple[float, float],
        debug_mlp: bool = True,
        # Per-layer neighbourhood parameters
        weight_kernels: Optional[list[list[int]]] = None,
        bias_kernels: Optional[list[list[int]]] = None,
        weight_strides: Optional[list[list[int]]] = None,
        bias_strides: Optional[list[list[int]]] = None,
        delta_abs: Optional[float] = None,
        # Early stopping
        early_stopping: bool = False,
        e_s_patience: int = 250,
        # Dynamic quantisation
        dynamic_quantization: bool = False,
        d_q_patience: int = 100,
        quantization_factor_multiplier: float = 10.0,
        max_quantization_factor: float = 1e4,
        # Dynamic kernel reshaping
        dynamic_kernel_reshaping: bool = False,
        d_k_r_patience: int = 100,
        x_weight_kernel_decr: int = 1,
        y_weight_kernel_decr: int = 1,
        y_bias_kernel_decr: int = 1,
        min_weight_kernel: Optional[list[int]] = None,  # Global minimum for all layers
        min_bias_kernel: Optional[list[int]] = None,
        x_stride_decr: int = 1,
        y_stride_decr: int = 1,
        min_x_stride: int = 1,
        min_y_stride: int = 1,
        # Thresholds
        loss_improvement_threshold: float = 1e-3,
        # General
        max_iterations: int = 1000,
        log_freq: int = 1000,
        measure_time: bool = True,
        save_trained_model: bool = False,
        model_name: str = "best_model",
    ) -> None:
        super().__init__(
            model=model,
            loss_fn=loss_fn,
            quantization_factor=quantization_factor,
            parameter_range=parameter_range,
            debug_mlp=debug_mlp,
            early_stopping=early_stopping,
            e_s_patience=e_s_patience,
            dynamic_quantization=dynamic_quantization,
            d_q_patience=d_q_patience,
            quantization_factor_multiplier=quantization_factor_multiplier,
            max_quantization_factor=max_quantization_factor,
            dynamic_kernel_reshaping=dynamic_kernel_reshaping,
            d_k_r_patience=d_k_r_patience,
            loss_improvement_threshold=loss_improvement_threshold,
            max_iterations=max_iterations,
            log_freq=log_freq,
            measure_time=measure_time,
            save_trained_model=save_trained_model,
            model_name=model_name,
        )

        # Deep-copy all per-layer lists to protect the caller's data from
        # in-place mutation during dynamic kernel reshaping.
        self.weight_kernels: list[list[int]] = (
            [list(k) for k in weight_kernels] if weight_kernels else [[2, 2]]
        )
        self.bias_kernels: list[list[int]] = (
            [list(k) for k in bias_kernels] if bias_kernels else [[2]]
        )
        self.weight_strides: list[list[int]] = (
            [list(s) for s in weight_strides] if weight_strides else [[1, 1]]
        )
        self.bias_strides: list[list[int]] = (
            [list(s) for s in bias_strides] if bias_strides else [[1]]
        )
        self.delta_abs = delta_abs

        # Kernel-reshaping configuration (global minimums applied to all layers)
        self.x_weight_kernel_decr = x_weight_kernel_decr
        self.y_weight_kernel_decr = y_weight_kernel_decr
        self.y_bias_kernel_decr = y_bias_kernel_decr
        self.min_weight_kernel: list[int] = (
            list(min_weight_kernel) if min_weight_kernel is not None else [1, 1]
        )
        self.min_bias_kernel: list[int] = (
            list(min_bias_kernel) if min_bias_kernel is not None else [1]
        )
        self.x_stride_decr = x_stride_decr
        self.y_stride_decr = y_stride_decr
        self.min_x_stride = min_x_stride
        self.min_y_stride = min_y_stride

    # ------------------------------------------------------------------
    # Abstract-method implementations
    # ------------------------------------------------------------------

    def _get_neighbors(
        self, node: SearchNode, X: Tensor, Y: Tensor
    ) -> NeighborList:
        return get_neighbors_layer_wise(
            node,
            X,
            Y,
            self.quantization_factor,
            weight_kernels=self.weight_kernels,
            bias_kernels=self.bias_kernels,
            weight_strides=self.weight_strides,
            bias_strides=self.bias_strides,
            delta_abs=self.delta_abs,
        )

    def _apply_kernel_reshape(self) -> bool:
        """Shrink every per-layer kernel and stride toward the global minimums."""
        modified = False

        for kernel in self.weight_kernels:
            if kernel[0] > self.min_weight_kernel[0]:
                kernel[0] = max(
                    kernel[0] - self.y_weight_kernel_decr,
                    self.min_weight_kernel[0],
                )
                modified = True
            if kernel[1] > self.min_weight_kernel[1]:
                kernel[1] = max(
                    kernel[1] - self.x_weight_kernel_decr,
                    self.min_weight_kernel[1],
                )
                modified = True

        for kernel in self.bias_kernels:
            if kernel[0] > self.min_bias_kernel[0]:
                kernel[0] = max(
                    kernel[0] - self.y_bias_kernel_decr,
                    self.min_bias_kernel[0],
                )
                modified = True

        for stride in self.weight_strides:
            if stride[0] > self.min_x_stride:
                stride[0] = max(stride[0] - self.x_stride_decr, self.min_x_stride)
                modified = True
            if stride[1] > self.min_y_stride:
                stride[1] = max(stride[1] - self.y_stride_decr, self.min_y_stride)
                modified = True

        # Bias strides are 1-D; their single component maps to the y-stride logic.
        for stride in self.bias_strides:
            if stride[0] > self.min_y_stride:
                stride[0] = max(stride[0] - self.y_stride_decr, self.min_y_stride)
                modified = True

        return modified

    def dynamic_reshape_kernels_and_strides(self) -> bool:
        """Shrink all per-layer kernels and strides toward their configured minima.

        .. deprecated::
            Call :meth:`_apply_kernel_reshape` directly in new code.  This
            public alias is retained for backward compatibility only.
        """
        return self._apply_kernel_reshape()


# ---------------------------------------------------------------------------
# TrainerRandomSampling  —  stochastic parameter perturbation
# ---------------------------------------------------------------------------


class TrainerRandomSampling(_BaseTrainer):
    """A*-search trainer using stochastic parameter perturbation.

    Uses :func:`get_neighbors_random` for neighbourhood generation.
    Neighbours are produced by randomly perturbing a fraction of the model's
    weights; no spatial kernel or stride is required.

    Dynamic kernel reshaping is **not** available for this trainer.

    Parameters that differ from :class:`Trainer`
    ---------------------------------------------
    perturbation_ratio:
        Fraction of parameters randomly perturbed when generating each
        neighbour.
    search_coverage_ratio:
        Fraction of the total parameter space explored per iteration.
    """

    def __init__(
        self,
        model: Module,
        loss_fn,
        quantization_factor: float,
        parameter_range: tuple[float, float],
        debug_mlp: bool = True,
        # Random-sampling neighbourhood parameters
        perturbation_ratio: float = 0.1,
        search_coverage_ratio: float = 0.5,
        delta_abs: Optional[float] = None,
        # Early stopping
        early_stopping: bool = False,
        e_s_patience: int = 250,
        # Dynamic quantisation
        dynamic_quantization: bool = False,
        d_q_patience: int = 100,
        quantization_factor_multiplier: float = 10.0,
        max_quantization_factor: float = 1e4,
        # Thresholds
        loss_improvement_threshold: float = 1e-3,
        # General
        max_iterations: int = 1000,
        log_freq: int = 1000,
        measure_time: bool = True,
        save_trained_model: bool = False,
        model_name: str = "best_model",
    ) -> None:
        super().__init__(
            model=model,
            loss_fn=loss_fn,
            quantization_factor=quantization_factor,
            parameter_range=parameter_range,
            debug_mlp=debug_mlp,
            early_stopping=early_stopping,
            e_s_patience=e_s_patience,
            dynamic_quantization=dynamic_quantization,
            d_q_patience=d_q_patience,
            quantization_factor_multiplier=quantization_factor_multiplier,
            max_quantization_factor=max_quantization_factor,
            dynamic_kernel_reshaping=False,  # Not supported by this trainer.
            loss_improvement_threshold=loss_improvement_threshold,
            max_iterations=max_iterations,
            log_freq=log_freq,
            measure_time=measure_time,
            save_trained_model=save_trained_model,
            model_name=model_name,
        )

        self.perturbation_ratio = perturbation_ratio
        self.search_coverage_ratio = search_coverage_ratio
        self.delta_abs = delta_abs

        # Narrow the log to only the features this trainer actually uses.
        self.dynamic_adjustments_log = {
            "dynamic_quantization_iterations": [],
        }

    # ------------------------------------------------------------------
    # Abstract-method implementation
    # ------------------------------------------------------------------

    def _get_neighbors(
        self, node: SearchNode, X: Tensor, Y: Tensor
    ) -> NeighborList:
        return get_neighbors_random(
            node,
            X,
            Y,
            self.quantization_factor,
            perturbation_ratio=self.perturbation_ratio,
            search_coverage_ratio=self.search_coverage_ratio,
            delta_abs=self.delta_abs,
        )
