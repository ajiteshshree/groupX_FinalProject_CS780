import multiprocessing as mp
import time

# Define a function to be executed by each process
def worker_function(name, delay):
    print(f"Worker {name} is starting...")
    time.sleep(delay)
    print(f"Worker {name} is finishing...")

if __name__ == '__main__':
    # Define the number of processes to create
    num_processes = 3
    
    # Create a list to hold the processes
    processes = []
    
    # Create and start multiple processes
    for i in range(1, num_processes + 1):
        p = mp.Process(target=worker_function, args=(i, i))
        processes.append(p)
        p.start()
    
    # Wait for all processes to finish
    for p in processes:
        p.join()
    
    # print(p)  
    print("All processes have finished.")
