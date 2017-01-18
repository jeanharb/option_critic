import matplotlib.pyplot as plt
import csv, pdb, sys
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pickle as pkl

def plot(filename, pdf_loc="training.pdf", csv_loc="training_progress.csv"):
  filename = "models/2016-01-15_14-36-05" if filename is None else filename
  open_loc = "%s/%s" % (filename, csv_loc)
  save_loc = "%s/%s" % (filename, pdf_loc)
  with open(open_loc) as csvfile:
    reader = csv.DictReader(csvfile)
    epochs = []
    scores = []
    q_vals = []
    for row in reader:
      epochs.append(row['epoch'])
      scores.append(float(row['mean_score']))
      q_vals.append(float(row['mean_q_val']))

  params = pkl.load(open("%s/%s" % (filename, 'model_params.pkl'), 'rb'))
  game_name = params.rom

  with open("/home/ml/jmerhe1/code/aleroms/rom_benchmark_scores.csv") as csvfile:
    #game,random,human,dqn,double_dqn
    reader = csv.DictReader(csvfile)
    for row in reader:
      if row['game'] == game_name:
        dqn_score = float(row['dqn'])
        double_dqn_score = float(row['double_dqn'])
        break

  smooth_scores = [np.mean(scores[max(0,i-10):i+1]) for i in range(len(scores))]
  fig, ax1 = plt.subplots()
  ax1.plot(epochs, scores, "r", label="testing score")
  ax1.plot(epochs, smooth_scores, "g", label="10 moving avg")
  spread = (max(scores+[dqn_score, double_dqn_score]) - min(scores+[dqn_score, double_dqn_score]))*1.1
  if spread > 0:
    ax1.set_ylim([max(scores+[dqn_score, double_dqn_score])-spread,min(scores+[dqn_score, double_dqn_score])+spread])


  ax1.axhline(dqn_score, c="y", label="DQN")
  ax1.axhline(double_dqn_score, c="m", label="Double DQN")
  box = ax1.get_position()
  ax1.set_position([box.x0, box.y0, box.width * 0.75, box.height])
  ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))

  """
  ax2 = ax1.twinx()
  ax2.plot(epochs, q_vals, 'b', label="avg q vals")
  spread = (max(q_vals) - min(q_vals))*1.1
  ax2.set_ylim([max(q_vals)-spread,min(q_vals)+spread])
  box = ax2.get_position()
  ax2.set_position([box.x0, box.y0, box.width * 0.75, box.height])
  ax2.legend(loc='center left', bbox_to_anchor=(1, 0.3))
  """
  #plt.show()
  pp = PdfPages(save_loc)
  pp.savefig(fig)
  pp.close()
  plt.close()

if __name__ == "__main__":
  plot(sys.argv[1] if len(sys.argv) > 1 else None)