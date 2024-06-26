import numpy as np
import visdom


class Dashboad():
    def __init__(self, port):

        self.vis = visdom.Visdom(port=port)

        self.trlosses = []
        self.vlLosses = []
        self.trAccs = []
        self.vlAccs = []
        self.epochs = []
        self.epochs1 = []

    def grid_res(self, images, nrow):
        self.vis.images(images, nrow=nrow, padding=10, opts=dict(title='Results'))

    def loss_curves(self, losses, epoch, win):

        opts = dict(xlabel='Epochs',
        ylabel='Loss',
        width=800, height=400,
        title='Train vs. Validation Loss',
        ytickmin=0, ytickmax=1,ytickstep=0.1, xtickmin=1,xtickmax=30, xtickstep=5,
        legend=['Train', 'Validation'])

        self.trlosses.append(losses[0])
        self.vlLosses.append(losses[1])
        self.epochs.append(epoch)

        X = np.column_stack((self.epochs, self.epochs))
        Y = np.column_stack((self.trlosses, self.vlLosses))

        if epoch == 1:
            win= self.vis.line(X=X, Y=Y, opts=opts)
        else:
            # self.vis.updateTrace(X=X, Y=Y, win=win, append=False)   # updateTrace已弃用
            self.vis.line(X=X, Y=Y, win=win, update='append')

        return win
    def acc_curves(self, accs, epoch, win):

        opts = dict(xlabel='Epochs',
        ylabel='Acc',
        width=800, height=400,
        title='Train vs. Validation Acc',
        ytickmin=0, ytickmax=1,ytickstep=0.1, xtickmin=1,xtickmax=30, xtickstep=5,
        legend=['Train', 'Validation'])

        self.trAccs.append(accs[0])
        self.vlAccs.append(accs[1])
        self.epochs1.append(epoch)

        X = np.column_stack((self.epochs1, self.epochs1))
        Y = np.column_stack((self.trAccs, self.vlAccs))
        if epoch == 1:
            win= self.vis.line(X=X, Y=Y, opts=opts)
        else:
            # self.vis.updateTrace(X=X, Y=Y, win=win, append=False)   # updateTrace已弃用
            self.vis.line(X=X, Y=Y, win=win, update='append')

        return win

    def metric_bar(self, X, title, nbins=10):
        self.vis.histogram(X=X, opts=dict(numbins=nbins, title=title))


