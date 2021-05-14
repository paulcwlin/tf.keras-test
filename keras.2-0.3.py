import util2 as u

history_dict = {'loss':[0.8,0.7,0.4,0.2,0.1],
                'acc':[0.2,0.3,0.5,0.7,0.9],
                'val_loss':[0.9,0.9,0.8,0.6,0.5],
                'val_acc':[0.1,0.2,0.6,0.7,0.8]}
u.plot(history_dict, ('loss', 'acc'),
       'Loss & Accuracy',
       ('Epochs', 'Loss & Acc'))