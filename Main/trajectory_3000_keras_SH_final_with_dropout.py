
'''
Editor : Mirae Sunny Kim
Update log

This is a deep learning algorithm for trajectory-based classifcation written with Keras packages.

no zero
MSD cut off at 5dt 0.001
uniform number of trajectories
individual without MCF10A

data reorganization: same percentage of each set in training/validation/testing

separate training and testing first, then kfold with training to get validation set
10/29/2019


Editor: Soonwoo
final version
comparision between dropout rate 0 and 0.4
01/04/2021

'''

version = '3000_final_with_dropout'

path = '/Users/Allen/Box Sync/EGFR_dynamics_deep_learning/kim_resnet'


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = "arial"
import keras
import os

from matplotlib.ticker import MaxNLocator

from keras.models import Sequential
from keras.utils import to_categorical, print_summary, plot_model
from keras.layers import Dense, Conv1D, MaxPooling1D, BatchNormalization, SpatialDropout1D
from keras.layers import Activation, GlobalAveragePooling1D, add

from keras.regularizers import l2
from keras import optimizers, Input, Model

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import roc_curve, auc

import umap

##########################################
# directory setting
##########################################

version_directory = path + "/" + version
if not os.path.isdir(version_directory + "/"):
    os.mkdir(version_directory + '/')
os.chdir(version_directory)

stat_directory = version_directory + "/stat"
if not os.path.isdir(stat_directory + "/"):
    os.mkdir(stat_directory + '/')

cm_directory = version_directory + "/confusion_matrix"
if not os.path.isdir(cm_directory + "/"):
    os.mkdir(cm_directory + '/')

umap_directory = version_directory + "/umap"
if not os.path.isdir(umap_directory + "/"):
    os.mkdir(umap_directory + '/')

model_directory = version_directory + "/model"
if not os.path.isdir(model_directory + "/"):
    os.mkdir(model_directory + '/')

data_directory = version_directory + "/data"
if not os.path.isdir(data_directory + "/"):
    os.mkdir(data_directory + '/')

##########################################
# parameters
##########################################

train_ratio = 0.8
max_length = 3000 # trajectory length
num_classes = 6
learning_rate = 5e-4
dropout_rate = 0.4
weight_decay = 5e-4

filter1 = 72
filter2 = 96

batch_size = 32
momentum = 0.01
folds = 5

eps = 1e-5
batch_decay = 0.99

patience = 50
epochs =5000

conv_kernel = 7
block_kernel = 5

font_size = 14

##########################################
# random seed
##########################################

np.random.seed(777)

##########################################
# model definition
##########################################

def get_model():

    model = Sequential()
    if 'model' in locals():
        model.reset_states()

    def identity_block(inputs, kernel_size, filters):

        filters1, filters2, filters3 = filters
        x = Conv1D(filters1, 1, use_bias=False, kernel_initializer='he_normal',
                   kernel_regularizer=l2(weight_decay))(inputs)
        x = BatchNormalization(momentum=batch_decay, epsilon = eps)(x)
        x = Activation('relu')(x)
        x = Conv1D(filters2, kernel_size, use_bias=False,
                   kernel_initializer='he_normal',
                   kernel_regularizer=l2(weight_decay),
                   padding = 'same')(x)
        x = BatchNormalization(momentum=batch_decay, epsilon = eps)(x)
        x = Activation('relu')(x)
        x = Conv1D(filters3, 1, use_bias=False,
                   kernel_initializer='he_normal',
                   kernel_regularizer=l2(weight_decay))(x)
        x = BatchNormalization(momentum=batch_decay, epsilon = eps)(x)
        x = add([x, inputs])
        x = Activation('relu')(x)
        return x


    def conv_block(inputs, kernel_size, filters, strides = 2):

        filters1, filters2, filters3 = filters
        x = Conv1D(filters1, 1, use_bias=False, kernel_initializer='he_normal',
                   kernel_regularizer=l2(weight_decay))(inputs)
        x = BatchNormalization(momentum=batch_decay, epsilon = eps)(x)
        x = Activation('relu')(x)
        x = Conv1D(filters2, kernel_size,
                   strides = strides,
                   use_bias=False,
                   kernel_initializer='he_normal',
                   kernel_regularizer=l2(weight_decay),
                   padding = 'same')(x)
        x = BatchNormalization(momentum=batch_decay, epsilon = eps)(x)
        x = Activation('relu')(x)
        x = Conv1D(filters3, 1, use_bias=False,
                   kernel_initializer='he_normal',
                   kernel_regularizer=l2(weight_decay))(x)
        x = BatchNormalization(momentum=batch_decay, epsilon = eps)(x)
        shortcut = Conv1D(filters3, 1, strides=strides, use_bias=False,
                          kernel_initializer='he_normal',
                          kernel_regularizer=l2(weight_decay))(inputs)
        shortcut = BatchNormalization(momentum=batch_decay, epsilon = eps)(shortcut)
        x = add([x, shortcut])
        x = Activation('relu')(x)
        return x

    inputs = Input(shape = (max_length,2))

    x = Conv1D(filter1, conv_kernel, strides = 2, padding = 'valid', use_bias=False, kernel_initializer='he_normal',
               kernel_regularizer=l2(weight_decay))(inputs)
    x = BatchNormalization(momentum=batch_decay, epsilon=eps)(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(3, strides = 2)(x)

    x = conv_block(x, block_kernel, [filter1,filter1,filter1*4])
    x = identity_block(x, block_kernel, [filter1, filter1, filter1*4])
    x = conv_block(x, block_kernel, [filter2,filter2,filter2*4])
    # x = identity_block(x,3, [filter2, filter2, filter2*4])

    x = SpatialDropout1D(rate = dropout_rate)(x)
    x = Conv1D(filter2, 11, strides = 2, padding = 'valid', use_bias=False, kernel_initializer='he_normal',
               kernel_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)

    x = GlobalAveragePooling1D()(x)
    '''
    # dense to 100 >> final layer >> dense to num_classes
    x = Dense(100, kernel_regularizer = l2(weight_decay),
              bias_regularizer = l2(weight_decay), name ='features_hundred')(x)
    final_layer = Model(inputs= inputs, outputs = x)
    x = Dense(num_classes, kernel_regularizer=l2(weight_decay),
              bias_regularizer=l2(weight_decay), name= 'features')(x)
    x = Activation('softmax')(x)
    '''
    x = Dense(num_classes, kernel_regularizer=l2(weight_decay),
              bias_regularizer=l2(weight_decay),name='features')(x)
    final_layer = Model(inputs = inputs, outputs = x)
    x = Activation('softmax')(x)


    model = Model(inputs=inputs, outputs=x)

    # optimizer
    sgd = optimizers.SGD(lr=learning_rate, momentum=momentum)
    adagrad = optimizers.Adagrad()
    adam = optimizers.Adam(lr=learning_rate)

    # compiler
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    return model, final_layer


def plot_confusion_matrix(cmatrix, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = cmatrix
    # Only use the labels that appear in the data
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap, vmin=0, vmax=1)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
    ax.set_xticklabels(classes, fontsize = font_size)
    ax.set_yticklabels(classes, fontsize = font_size)
    ax.set_xlabel('Predicted label', fontsize = font_size)
    ax.set_ylabel('True label', fontsize = font_size)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    plt.title(title, fontsize=font_size)
    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = 0.6
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize = font_size)
    fig.tight_layout()
    return ax

##########################################
# Load Data
##########################################

data_path = '/Users/Allen/Box Sync/EGFR_dynamics_deep_learning/data'

data_type = 'woMCF10A_uneven'

# load data
data_x = np.loadtxt(path+'/data_'+str(max_length)+'_x_'+data_type+'.csv', delimiter = ',')
data_y = np.loadtxt(path+'/data_'+str(max_length)+'_y_'+data_type+'.csv', delimiter = ',')
data_label = np.loadtxt(path+'/data_'+str(max_length)+'_label_'+data_type+'.csv', delimiter = ',')

data_x = np.reshape(data_x, (-1, max_length, 1))
data_y = np.reshape(data_y, (-1, max_length, 1))
X = np.concatenate((data_x, data_y), axis=2)

data_type = 'EMT_uneven'

# load data
emt_x_coord = np.loadtxt(path+'/data_'+str(max_length)+'_x_'+data_type+'.csv', delimiter = ',')
emt_y_coord = np.loadtxt(path+'/data_'+str(max_length)+'_y_'+data_type+'.csv', delimiter = ',')
emt_label = np.loadtxt(path+'/data_'+str(max_length)+'_label_'+data_type+'.csv', delimiter = ',')

emt_x_coord = np.reshape(emt_x_coord, (-1, max_length, 1))
emt_y_coord = np.reshape(emt_y_coord, (-1, max_length, 1))
emt_x = np.concatenate((emt_x_coord, emt_y_coord), axis=2)
emt_ground = emt_label

# split data into folds
skf = StratifiedKFold(n_splits= folds, shuffle=True)

# early stopping options. Stop training (patience) steps after val_loss starts to increase.
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',
                                               patience=patience, verbose=1,
                                               mode='min',
                                               baseline=None,
                                               restore_best_weights=True)

##########################################
# Folds
##########################################
cvscores = []

# blank matrix
train_acc = []
train_loss = []

val_acc = []
val_loss = []

test_loss =[]
test_acc = []

iteration = 1
first = True

emt_naming = "_lr_" + str(learning_rate) + "_wd_" + str(weight_decay) + "_mo_" + str(
        momentum) + "_emt_" + str(len(emt_x)) + "_numfolds_" + str(folds)


for train, test in skf.split(X, data_label):
    train_idx = train
    test_idx = test

test_x = X[test_idx]
test_y = data_label[test_idx]
test_y = to_categorical(test_y,num_classes)

X = X[train_idx]
Y = data_label[train_idx]

for train_index, valid_index in skf.split(X, Y):

    fine_tuning_naming = "_lr_" + str(learning_rate) + "_wd_" + str(weight_decay) + "_mo_" + str(
        momentum) + "_train_" + str(len(train_index)) + "_valid_" + str(len(valid_index)) + "_fold_" + str(iteration)

    y = to_categorical(Y, num_classes)
    train_x = X[train_index]
    train_y = y[train_index]

    valid_x = X[valid_index]
    valid_y = y[valid_index]

    # model
    model, final_layer = get_model()

    ########################################
    # training
    ########################################

    history = model.fit(x=train_x, y=train_y,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=0,
                        validation_data = (valid_x, valid_y),
                        shuffle=True
                        , callbacks=[early_stopping]
                        )

    ########################################
    # evaluation
    ########################################

    eval = model.evaluate(x = test_x, y = test_y, verbose = 1)
    print(eval)
    print(model.metrics_names)

    ########################################
    # visualization
    ########################################

    #f = open('model_parameters.txt', 'w')
    #sys.stdout = f
    #print_summary(model)

    plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

    # Plot training & validation accuracy values
    ax1 = plt.figure().gca()
    ax1.plot(history.history['acc'])
    ax1.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    plt.ylim(0, 1)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.savefig(stat_directory + '/accuracy'+ fine_tuning_naming + '.png', dpi=300)
    print()
    plt.close()

    # Plot training & validation loss values
    ax2 = plt.figure().gca()
    ax2.plot(history.history['loss'])
    ax2.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.savefig(stat_directory + '/loss' + fine_tuning_naming +'.png', dpi=300)
    print()
    plt.close()

    # save loss/acc per each set
    train_loss.append(history.history['loss'])
    val_loss.append(history.history['val_loss'])
    train_acc.append(history.history['acc'])
    val_acc.append(history.history['val_acc'])

    test_evaluation = model.evaluate(x=test_x, y=test_y, batch_size=batch_size, verbose=0)
    test_loss.append(test_evaluation[0])
    test_acc.append(test_evaluation[1])

    if first==True:
        prediction = model.predict(x=test_x, batch_size=batch_size, verbose=0)
        prediction0 = np.expand_dims(prediction, axis=0)
        test_pred = prediction0

        embedding_ing0 = final_layer.predict(x=test_x, batch_size=batch_size, verbose=0)
        embedding_ing0 = np.expand_dims(embedding_ing0, axis=0)
        embedding = embedding_ing0

        emt_prediction0 = model.predict(x=emt_x, batch_size=batch_size, verbose=0)
        emt_prediction0 = np.expand_dims(emt_prediction0, axis=0)
        emt_pred = emt_prediction0

        emt_embedding_ing0 = final_layer.predict(x=emt_x, batch_size=batch_size, verbose=0)
        emt_embedding_ing0 = np.expand_dims(emt_embedding_ing0, axis=0)
        emt_embedding = emt_embedding_ing0

        first = False
    else:
        prediction = model.predict(x=test_x, batch_size=batch_size, verbose=0)
        prediction = np.expand_dims(prediction, axis=0)
        test_pred = np.concatenate((test_pred, prediction), axis=0)

        embedding_ing = final_layer.predict(x=test_x, batch_size=batch_size, verbose=0)
        embedding_ing = np.expand_dims(embedding_ing, axis=0)
        embedding = np.concatenate((embedding, embedding_ing), axis=0)

        emt_prediction = model.predict(x=emt_x, batch_size=batch_size, verbose=0)
        emt_prediction = np.expand_dims(emt_prediction, axis=0)
        emt_pred = emt_prediction

        emt_embedding_ing = final_layer.predict(x=emt_x, batch_size=batch_size, verbose=0)
        emt_embedding_ing = np.expand_dims(emt_embedding_ing, axis=0)
        emt_embedding = emt_embedding_ing

    ########################################
    # save results
    ########################################

    # model_save
    modelname = 'modelsave_' + version + '_fold_' + str(iteration) + '.h5'
    iteration = iteration + 1
    # remove redundant files from before (does not create files if same name exists already)
    if os.path.isfile(model_directory + '/' + modelname):
        os.remove(model_directory + '/' + modelname)
    if os.path.isfile(path + '/' + modelname):
        os.remove(path + '/' + modelname)

    # save in current version directory
    model.save(modelname)

    # move it to model directory
    os.rename(version_directory + '/' + modelname, model_directory + '/' + modelname)
'''
    ######################################
    # individual confusion matrix
    ######################################

    current_test_pred_argmax = np.argmax(prediction, axis=1)
    current_test_ground = np.argmax(test_y, axis=1)

    current_cmatrix = confusion_matrix(current_test_ground, current_test_pred_argmax)

    # enter labels
    class_name_list = ['MCF7',
                       'BT474',
                       'SKBR3',
                       'MDA-MB-468',
                       'MDA-MB-231',
                       'BT549']

    colors = ['r',
              'g',
              'b',
              'c',
              'm',
              'y']

    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plot_confusion_matrix(current_cmatrix, classes=class_name_list,
                          normalize=True)
    plt.savefig(version_directory + '/confusion_matrix' + fine_tuning_naming + '.png', dpi=300)
'''
test_pred_sum = np.sum(test_pred, axis=0)
test_pred_argmax = np.argmax(test_pred_sum, axis=1)
test_ground = np.argmax(test_y, axis=1)
total_test_accuracy = accuracy_score(test_ground, test_pred_argmax)

emt_pred_sum = np.sum(emt_pred, axis=0)
emt_pred_argmax = np.argmax(emt_pred_sum, axis=1)

# test accuracy save
acc_text = open(version_directory + "/test_accuracy.txt", "w")
acc_text.write("This is k-fold accuracy over test dataset\n")
for i in range(folds):
    acc_text.write(str(100 * round(test_acc[i], 4)) + " %\n")
acc_text.write("This is total accuracy over test dataset\n")
acc_text.write(str(100 * round(total_test_accuracy, 4)) + " %\n")
acc_text.close()

embedding_sum = np.sum(embedding, axis=0)
emt_embedding_sum = np.sum(emt_embedding, axis=0)

####################################
# testing confusion matrix
####################################

cmatrix = confusion_matrix(test_ground, test_pred_argmax)

# enter labels
class_name_list = ['MCF7',
                   'BT474',
                   'SKBR3',
                   'MDA-MB-468',
                   'MDA-MB-231',
                   'BT549']

colors = ['r',
          'g',
          'b',
          'c',
          'm',
          'y']

np.set_printoptions(precision=2)

# Plot normalized confusion matrix
np.savetxt(data_directory+"/cmatrix.npy", cmatrix)
plot_confusion_matrix(cmatrix, classes=class_name_list,
                      normalize=True)
plt.savefig(version_directory + '/confusion_matrix' + fine_tuning_naming + '.png', dpi=300)

####################################
# emt confusion matrix
####################################
'''
cmatrix = confusion_matrix(emt_ground, emt_pred_argmax)

# enter labels
emt_labels  = ['MCF10A EMT',
               'MCF10A',
               'MCF7 EMT',
               'MCF7',
               'MDAMB231 EMT',
               'MDAMB231']

np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
cmatrix = cmatrix[0:6,0:6]
emt_confusion_matrix(cmatrix, classes=class_name_list,
                      normalize=True)
plt.savefig(version_directory + '/confusion_matrix' + emt_naming + '.png', dpi=300)
'''
####################################
# umap
####################################

np.savetxt(data_directory + '/embedding.npy', embedding_sum)
np.savetxt(data_directory + '/testground.npy', test_ground)

reducer = umap.UMAP(
    #n_neighbors = int((len(test_x)/num_classes)*total_test_accuracy)
)
lowDWeights = reducer.fit_transform(embedding_sum)
umap_fig, umap_ax = plt.subplots(figsize=(7, 7))
for i in range(len(class_name_list)):
    class_idx = np.where(test_ground == i)
    umap_ax.scatter(lowDWeights[class_idx, 0], lowDWeights[class_idx, 1], c=colors[i], label=class_name_list[i])

umap_ax.legend(bbox_to_anchor=(1.02, 0.5), loc='center left', frameon=False, fontsize='14')
umap_fig.tight_layout()
umap_ax.set_aspect('equal', 'box')
umap_ax.axis('off')
umap_ax.margins(0.06)
plt.savefig(umap_directory + '/' + 'umap.png', dpi=300, bbox_inches='tight')
plt.close()

plot_model(model, to_file=version_directory+'/model.png', show_shapes=True, show_layer_names=True)



#########################################
# roc curve
#########################################
roc_curve_plot = True

if roc_curve_plot:

    roc_directory = version_directory + "/" + "roc"
    if not os.path.isdir(roc_directory + "/"):
        os.mkdir(roc_directory + '/')

    lw = 2
    fs = 14

    fpr = dict()  # 1-specificity
    tpr = dict()  # sensitivity
    roc_auc = dict()
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(test_y[:, i], test_pred_sum[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    colors = ['grey',
              'r',
              'g',
              'b',
              'y',
              'c']  # color for each label

    all_roc_fig, all_roc_ax = plt.subplots()

    all_roc_ax.hlines(0.8, -0.03, 1.03, linestyles='dashed', lw=lw - 1)
    all_roc_ax.hlines(0.9, -0.03, 1.03, linestyles='dashed', lw=lw - 1)
    all_roc_ax.hlines(1.0, -0.03, 1.03, linestyles='dashed', lw=lw - 1)

    all_roc_ax.vlines(0.0, 0, 1.03, linestyles='dashed', lw=lw - 1)
    all_roc_ax.vlines(0.1, 0, 1.03, linestyles='dashed', lw=lw - 1)
    all_roc_ax.vlines(0.2, 0, 1.03, linestyles='dashed', lw=lw - 1)

    for i, color in zip(range(num_classes), colors):
        all_roc_im = all_roc_ax.plot((fpr[i]), tpr[i], color=color, lw=lw,
                                     label=class_name_list[i] + ' (AUC = {0:0.2f})'.format(roc_auc[i]))
    all_roc_ax.legend(loc="lower right", frameon=False)
    all_roc_ax.set_ylabel("Sensitivity", fontsize=fs)
    all_roc_ax.set_xlabel("1-Specificity", fontsize=fs)
    all_roc_ax.set_aspect('equal', 'box')
    all_roc_ax.set(xlim=(-0.03, 1.03), ylim=(0, 1.03))
    # all_roc_ax.set_xlim([0.0, 1.03])
    # all_roc_ax.set_ylim([0.0, 1.03])
    all_roc_ax.spines['top'].set_visible(False)
    all_roc_ax.spines['right'].set_visible(False)
    all_roc_ax.spines['left'].set_linewidth(lw - .5)
    all_roc_ax.spines['bottom'].set_linewidth(lw - .5)
    all_roc_fig.tight_layout()

    plt.savefig(roc_directory + '/roc_total' + '.png', dpi=600)
    plt.close()

    # separate ROC curve

    for i in range(num_classes):
        roc_fig, roc_ax = plt.subplots()
        roc_ax.hlines(0.8, -0.03, 1.03, linestyles='dashed', lw=lw - 1)
        roc_ax.hlines(0.9, -0.03, 1.03, linestyles='dashed', lw=lw - 1)
        roc_ax.hlines(1.0, -0.03, 1.03, linestyles='dashed', lw=lw - 1)
        roc_ax.vlines(0.0, 0, 1.03, linestyles='dashed', lw=lw - 1)
        roc_ax.vlines(0.1, 0, 1.03, linestyles='dashed', lw=lw - 1)
        roc_ax.vlines(0.2, 0, 1.03, linestyles='dashed', lw=lw - 1)
        roc_im = roc_ax.plot(fpr[i], tpr[i], color='b', lw=lw, label='AUC = {0:0.2f}'.format(roc_auc[i]))
        roc_ax.set_ylabel("Sensitivity", fontsize=fs)
        roc_ax.set_xlabel("1-Specificity", fontsize=fs)
        roc_ax.set_title(class_name_list[i], fontsize=fs)
        roc_ax.legend(loc="lower right", frameon=False)

        roc_ax.set_aspect('equal', 'box')
        roc_ax.set(xlim=(-0.03, 1.03), ylim=(0, 1.03))
        roc_ax.spines['top'].set_visible(False)
        roc_ax.spines['right'].set_visible(False)
        roc_ax.spines['left'].set_linewidth(lw - .5)
        roc_ax.spines['bottom'].set_linewidth(lw - .5)
        roc_fig.tight_layout()
        plt.savefig(roc_directory + '/' + class_name_list[i] + '.png', dpi=600)
        plt.close()

#########################################
# activate tensorboard
#########################################
tb_hist = tf.keras.callbacks.TensorBoard(log_dir='./logs')