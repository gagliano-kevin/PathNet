import psutil


class SystemMemoryGuard:
    def __init__(self, threshold_percentage=90.0):
        self.threshold = threshold_percentage

    def memory_exceeded(self):
        # Get the percentage of total RAM used across the whole system
        stats = psutil.virtual_memory()
        total_usage_percent = stats.percent
        
        # Convert to Megabytes for readability
        mem_used_mb = stats.used / (1024**2)
        mem_available_mb = stats.available / (1024**2)
        mem_total_mb = stats.total / (1024**2)
        
        if total_usage_percent > self.threshold:
            print("\n" + "="*40)
            print("       [!] MEMORY GUARD ALERT        ")
            print("="*40)
            print(f"STATUS:    EXCEEDED ({total_usage_percent}%)")
            print(f"LIMIT:     {self.threshold}%")
            print("-"*40)
            print(f"USED:      {mem_used_mb:10.2f} MB")
            print(f"AVAILABLE: {mem_available_mb:10.2f} MB")
            print(f"TOTAL:     {mem_total_mb:10.2f} MB")
            print("="*40 + "\n")
            
            # Return True if memory usage exceeded threshold
            return True
        
        return False