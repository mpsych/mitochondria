from imports import *

class InteractivePlot(keras.callbacks.Callback):
    def __init__(self, logfile, doplot, num_batches):
        self.logfile = logfile
        self.doplot = doplot
        self.f = open(logfile, 'a', 1)
        self.num_batches = num_batches

    def on_train_begin(self, logs={}):
        print("starting...")
        self.losses = []
        self.acc = []
        self.prec = []
        self.rec = []
        self.jaccard = []
        self.dice = []

        self.losses_val = []
        self.acc_val = []
        self.prec_val = []
        self.rec_val = []
        self.jaccard_val = []
        self.dice_val = []

        self.batchnr = 0
        self.time_epoch = []
        self.timestamp = time.time()
        self.icount = 0

        self.logs = []

    def on_train_end(self, logs={}):
        self.f.close()
        print('training ended, logfile closed.')

    def on_epoch_end(self, epoch, logs={}):
        self.batchnr = 0
        elapsed = time.time() - self.timestamp
        self.timestamp = time.time()
        self.time_epoch.append(elapsed)

        loss_train = logs.get('loss')
        acc_train = logs.get('acc')
        prec = logs.get('keras_precision')
        recall = logs.get('keras_recall')
        jaccard = logs.get('keras_jaccard_coef')
        dice = logs.get('keras_dice_coef')

        loss_val = logs.get('val_loss')
        acc_val = logs.get('val_acc')
        prec_val = logs.get('val_keras_precision')
        recall_val = logs.get('val_keras_recall')
        jaccard_val = logs.get('val_keras_jaccard_coef')
        dice_val = logs.get('val_keras_dice_coef')


        self.losses.append(loss_train)
        self.acc.append(acc_train)
        self.prec.append(prec)
        self.rec.append(recall)
        self.jaccard.append(jaccard)
        self.dice.append(dice)

        self.losses_val.append(loss_val)
        self.acc_val.append(acc_val)
        self.prec_val.append(prec_val)
        self.rec_val.append(recall_val)
        self.jaccard_val.append(jaccard_val)
        self.dice_val.append(dice_val)

        self.icount+=1        

        if self.doplot:
            clear_output(wait=True)
            plt.figure(figsize=(14,10))
            train_vals = [self.losses, self.acc, self.prec, self.rec, self.jaccard, self.dice]
            test_vals = [self.losses_val, self.acc_val, self.prec_val, self.rec_val, self.jaccard_val, self.dice_val]
            desc = ['loss', 'accuracy', 'precision', 'recall', 'jaccard', 'dice']
            for i in range(len(train_vals)):
                plt.subplot(2, 3, i+1)
                plt.plot(range(self.icount), train_vals[i], label=desc[i])
                plt.plot(range(self.icount), test_vals[i], label="val_" + desc[i])
                plt.ylim(0,1)
                plt.legend()
            plt.savefig(self.logfile.replace('.txt', '.png'), bbox_inches='tight', format='png')   
            plt.show()

        else:
            clear_output(wait=True)

        print('iteration', self.icount)
        print('loss', loss_train, 'loss_val', loss_val)
        print('acc', acc_train, 'acc_val', acc_val)
        print('prec', prec, 'prec_val', prec_val)
        print('rec', recall, 'rec_val', recall_val)
        print('jaccard', jaccard, 'jaccard_val', jaccard_val)
        print('dice', dice, 'dice_val', dice_val)
        print('time per epoch:', np.array(self.time_epoch).mean())
        self.f.write('iteration=' + str(self.icount) + '|loss_val=' + str(loss_val) + '|acc_val=' + str(acc_val) + '|prec_val=' + str(prec_val) + '|rec_val=' + str(recall_val) + '|jaccard_val=' + str(jaccard_val) + '|dice_val=' + str(dice_val) + '|loss=' + str(loss_train) + '|acc=' + str(acc_train) + '|prec=' + str(prec) + '|rec=' + str(recall) + '|jaccard=' + str(jaccard) + '|dice=' + str(dice) + '|time=' + str(elapsed) + '\n')

    def on_batch_end(self, batch, logs=None):  
        self.batchnr+=1
        if self.batchnr%10==0:
            print('batch', self.batchnr, 'of', self.num_batches, '...')

