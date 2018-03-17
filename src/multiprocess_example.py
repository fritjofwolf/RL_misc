import multiprocessing as mp
import string
import time

startzeit = time.time()
# Define an output queue
output = mp.Queue()

# define a example function
def rand_string(length, output):
    """ Generates a random string of numbers, lower- and uppercase chars. """
    a = 0
    for i in range(10**8):
        a += 1
    output.put(a)

# Setup a list of processes that we want to run
processes = [mp.Process(target=rand_string, args=(5, output)) for x in range(4)]

# Run processes
for p in processes:
    p.start()

# Exit the completed processes
for p in processes:
    p.join()

# Get process results from the output queue
results = [output.get() for p in processes]
stopzeit = time.time()

print("Zeitdauer für Multithreads: ", stopzeit-startzeit)

startzeit = time.time()
output = mp.Queue()
for i in range(4):
    rand_string(5, output)
stopzeit = time.time()
print("Zeitdauer für Singlethread: ", stopzeit-startzeit)