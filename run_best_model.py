import subprocess, sys

def run_watch():
  command = ['python', 'train_q.py', '--steps-per-epoch', '0',
  '--test-length', '100000', '--nn-file', sys.argv[1], '--display-screen',
  '--max-history', '10', '--testing'] + sys.argv[2:]

  p1 = subprocess.Popen(command)
  p1.wait()
    
if __name__ == "__main__":
  run_watch()
