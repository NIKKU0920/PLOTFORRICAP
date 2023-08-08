import numpy as np
import matplotlib.pyplot as plt

#############################
######CIFAR10 BUDGET0.2######
#############################

#ACC_TRAIN
Accuracy_train1 = np.load("/home/nick/CL14/ImportanceSampling-ex14/metrics_unif-SGD_cifar10_ricap_b02s2/accuracy_per_epoch_train.npy")
Accuracy_train2 = np.load("/home/nick/CL15/ImportanceSampling-ex15/metrics_unif-SGD_cifar10_ricap_b02s2/accuracy_per_epoch_train.npy")
Accuracy_train3 = np.load("/home/nick/CL16/ImportanceSampling-ex16/metrics_unif-SGD_cifar10_ricap_b02s2/accuracy_per_epoch_train.npy")
Accuracy_train4 = np.load("/home/nick/CL17/ImportanceSampling-ex17/metrics_unif-SGD_cifar10_ricap_b02s2/accuracy_per_epoch_train.npy")
# Plot each line with a different color
plt.plot(Accuracy_train1, color='blue', label='Case 1')
plt.plot(Accuracy_train2, color='red', label='Case 2')
plt.plot(Accuracy_train3, color='green', label='Case 3')
plt.plot(Accuracy_train4, color='yellow', label='Case 4')
plt.xlabel('Epoch')
plt.ylabel('Train accuracy')
plt.title('Train accuracy per Epoch under CIFAR-10 & budget=0.2')
# Show the legend with the labels
plt.legend()
# Optionally display the plot
plt.show()
# Save the plot to a file (e.g., PNG)
plt.savefig("/home/nick/PLOTFORRICAP/resultsforacctrain_c10b2.png")

# Reset the figure to create a new independent graph for test accuracy
plt.figure()

#ACC_TEST
Accuracy_test1 = np.load("/home/nick/CL14/ImportanceSampling-ex14/metrics_unif-SGD_cifar10_ricap_b02s2/accuracy_per_epoch_val.npy")
Accuracy_test2 = np.load("/home/nick/CL15/ImportanceSampling-ex15/metrics_unif-SGD_cifar10_ricap_b02s2/accuracy_per_epoch_val.npy")
Accuracy_test3 = np.load("/home/nick/CL16/ImportanceSampling-ex16/metrics_unif-SGD_cifar10_ricap_b02s2/accuracy_per_epoch_val.npy")
Accuracy_test4 = np.load("/home/nick/CL17/ImportanceSampling-ex17/metrics_unif-SGD_cifar10_ricap_b02s2/accuracy_per_epoch_val.npy")
# Plot each line with a different color
plt.plot(Accuracy_test1, color='blue', label='Case 1')
plt.plot(Accuracy_test2, color='red', label='Case 2')
plt.plot(Accuracy_test3, color='green', label='Case 3')
plt.plot(Accuracy_test4, color='yellow', label='Case 4')
plt.xlabel('Epoch')
plt.ylabel('Test accuracy')
plt.title('Test accuracy per Epoch under CIFAR-10 & budget=0.2')
# Show the legend with the labels
plt.legend()
# Optionally display the plot
plt.show()
# Save the plot to a file (e.g., PNG)
plt.savefig("/home/nick/PLOTFORRICAP/resultsforacctest_c10b2.png")

# Reset the figure to create a new independent graph for test accuracy
plt.figure()

#LOSS_TRAIN
Loss_train1 = np.load("/home/nick/CL14/ImportanceSampling-ex14/metrics_unif-SGD_cifar10_ricap_b02s2/LOSS_epoch_train.npy")
Loss_train2 = np.load("/home/nick/CL15/ImportanceSampling-ex15/metrics_unif-SGD_cifar10_ricap_b02s2/LOSS_epoch_train.npy")
Loss_train3 = np.load("/home/nick/CL16/ImportanceSampling-ex16/metrics_unif-SGD_cifar10_ricap_b02s2/LOSS_epoch_train.npy")
Loss_train4 = np.load("/home/nick/CL17/ImportanceSampling-ex17/metrics_unif-SGD_cifar10_ricap_b02s2/LOSS_epoch_train.npy")
# Plot each line with a different color
plt.plot(Loss_train1, color='blue', label='Case 1')
plt.plot(Loss_train2, color='red', label='Case 2')
plt.plot(Loss_train3, color='green', label='Case 3')
plt.plot(Loss_train4, color='yellow', label='Case 4')
plt.xlabel('Epoch')
plt.ylabel('Train loss')
plt.title('Train loss per Epoch under CIFAR-10 & budget=0.2')
# Show the legend with the labels
plt.legend()
# Optionally display the plot
plt.show()
# Save the plot to a file (e.g., PNG)
plt.savefig("/home/nick/PLOTFORRICAP/resultsforlosstrain_c10b2.png")

# Reset the figure to create a new independent graph for test accuracy
plt.figure()

#LOSS_TEST
Loss_test1 = np.load("/home/nick/CL14/ImportanceSampling-ex14/metrics_unif-SGD_cifar10_ricap_b02s2/LOSS_epoch_val.npy")
Loss_test2 = np.load("/home/nick/CL15/ImportanceSampling-ex15/metrics_unif-SGD_cifar10_ricap_b02s2/LOSS_epoch_val.npy")
Loss_test3 = np.load("/home/nick/CL16/ImportanceSampling-ex16/metrics_unif-SGD_cifar10_ricap_b02s2/LOSS_epoch_val.npy")
Loss_test4 = np.load("/home/nick/CL17/ImportanceSampling-ex17/metrics_unif-SGD_cifar10_ricap_b02s2/LOSS_epoch_val.npy")
# Plot each line with a different color
plt.plot(Loss_test1, color='blue', label='Case 1')
plt.plot(Loss_test2, color='red', label='Case 2')
plt.plot(Loss_test3, color='green', label='Case 3')
plt.plot(Loss_test4, color='yellow', label='Case 4')
plt.xlabel('Epoch')
plt.ylabel('Test loss')
plt.title('Test loss per Epoch under CIFAR-10 & budget=0.2')
# Show the legend with the labels
plt.legend()
# Optionally display the plot
plt.show()
# Save the plot to a file (e.g., PNG)
plt.savefig("/home/nick/PLOTFORRICAP/resultsforlosstest_c10b2.png")

# Reset the figure to create a new independent graph for test accuracy
plt.figure()



#############################
######CIFAR10 BUDGET0.3######
#############################

#ACC_TRAIN
Accuracy_train1 = np.load("/home/nick/CL14/ImportanceSampling-ex14/metrics_unif-SGD_cifar10_ricap_b03s2/accuracy_per_epoch_train.npy")
Accuracy_train2 = np.load("/home/nick/CL15/ImportanceSampling-ex15/metrics_unif-SGD_cifar10_ricap_b03s2/accuracy_per_epoch_train.npy")
Accuracy_train3 = np.load("/home/nick/CL16/ImportanceSampling-ex16/metrics_unif-SGD_cifar10_ricap_b03s2/accuracy_per_epoch_train.npy")
Accuracy_train4 = np.load("/home/nick/CL17/ImportanceSampling-ex17/metrics_unif-SGD_cifar10_ricap_b03s2/accuracy_per_epoch_train.npy")
# Plot each line with a different color
plt.plot(Accuracy_train1, color='blue', label='Case 1')
plt.plot(Accuracy_train2, color='red', label='Case 2')
plt.plot(Accuracy_train3, color='green', label='Case 3')
plt.plot(Accuracy_train4, color='yellow', label='Case 4')
plt.xlabel('Epoch')
plt.ylabel('Train accuracy')
plt.title('Train accuracy per Epoch under CIFAR-10 & budget=0.3')
# Show the legend with the labels
plt.legend()
# Optionally display the plot
plt.show()
# Save the plot to a file (e.g., PNG)
plt.savefig("/home/nick/PLOTFORRICAP/resultsforacctrain_c10b3.png")

# Reset the figure to create a new independent graph for test accuracy
plt.figure()

#ACC_TEST
Accuracy_test1 = np.load("/home/nick/CL14/ImportanceSampling-ex14/metrics_unif-SGD_cifar10_ricap_b03s2/accuracy_per_epoch_val.npy")
Accuracy_test2 = np.load("/home/nick/CL15/ImportanceSampling-ex15/metrics_unif-SGD_cifar10_ricap_b03s2/accuracy_per_epoch_val.npy")
Accuracy_test3 = np.load("/home/nick/CL16/ImportanceSampling-ex16/metrics_unif-SGD_cifar10_ricap_b03s2/accuracy_per_epoch_val.npy")
Accuracy_test4 = np.load("/home/nick/CL17/ImportanceSampling-ex17/metrics_unif-SGD_cifar10_ricap_b03s2/accuracy_per_epoch_val.npy")
# Plot each line with a different color
plt.plot(Accuracy_test1, color='blue', label='Case 1')
plt.plot(Accuracy_test2, color='red', label='Case 2')
plt.plot(Accuracy_test3, color='green', label='Case 3')
plt.plot(Accuracy_test4, color='yellow', label='Case 4')
plt.xlabel('Epoch')
plt.ylabel('Test accuracy')
plt.title('Test accuracy per Epoch under CIFAR-10 & budget=0.3')
# Show the legend with the labels
plt.legend()
# Optionally display the plot
plt.show()
# Save the plot to a file (e.g., PNG)
plt.savefig("/home/nick/PLOTFORRICAP/resultsforacctest_c10b3.png")

# Reset the figure to create a new independent graph for test accuracy
plt.figure()

#LOSS_TRAIN
Loss_train1 = np.load("/home/nick/CL14/ImportanceSampling-ex14/metrics_unif-SGD_cifar10_ricap_b03s2/LOSS_epoch_train.npy")
Loss_train2 = np.load("/home/nick/CL15/ImportanceSampling-ex15/metrics_unif-SGD_cifar10_ricap_b03s2/LOSS_epoch_train.npy")
Loss_train3 = np.load("/home/nick/CL16/ImportanceSampling-ex16/metrics_unif-SGD_cifar10_ricap_b03s2/LOSS_epoch_train.npy")
Loss_train4 = np.load("/home/nick/CL17/ImportanceSampling-ex17/metrics_unif-SGD_cifar10_ricap_b03s2/LOSS_epoch_train.npy")
# Plot each line with a different color
plt.plot(Loss_train1, color='blue', label='Case 1')
plt.plot(Loss_train2, color='red', label='Case 2')
plt.plot(Loss_train3, color='green', label='Case 3')
plt.plot(Loss_train4, color='yellow', label='Case 4')
plt.xlabel('Epoch')
plt.ylabel('Train loss')
plt.title('Train loss per Epoch under CIFAR-10 & budget=0.3')
# Show the legend with the labels
plt.legend()
# Optionally display the plot
plt.show()
# Save the plot to a file (e.g., PNG)
plt.savefig("/home/nick/PLOTFORRICAP/resultsforlosstrain_c10b3.png")

# Reset the figure to create a new independent graph for test accuracy
plt.figure()

#LOSS_TEST
Loss_test1 = np.load("/home/nick/CL14/ImportanceSampling-ex14/metrics_unif-SGD_cifar10_ricap_b03s2/LOSS_epoch_val.npy")
Loss_test2 = np.load("/home/nick/CL15/ImportanceSampling-ex15/metrics_unif-SGD_cifar10_ricap_b03s2/LOSS_epoch_val.npy")
Loss_test3 = np.load("/home/nick/CL16/ImportanceSampling-ex16/metrics_unif-SGD_cifar10_ricap_b03s2/LOSS_epoch_val.npy")
Loss_test4 = np.load("/home/nick/CL17/ImportanceSampling-ex17/metrics_unif-SGD_cifar10_ricap_b03s2/LOSS_epoch_val.npy")
# Plot each line with a different color
plt.plot(Loss_test1, color='blue', label='Case 1')
plt.plot(Loss_test2, color='red', label='Case 2')
plt.plot(Loss_test3, color='green', label='Case 3')
plt.plot(Loss_test4, color='yellow', label='Case 4')
plt.xlabel('Epoch')
plt.ylabel('Test loss')
plt.title('Test loss per Epoch under CIFAR-10 & budget=0.3')
# Show the legend with the labels
plt.legend()
# Optionally display the plot
plt.show()
# Save the plot to a file (e.g., PNG)
plt.savefig("/home/nick/PLOTFORRICAP/resultsforlosstest_c10b3.png")

# Reset the figure to create a new independent graph for test accuracy
plt.figure()


#############################
######CIFAR10 BUDGET0.5######
#############################

#ACC_TRAIN
Accuracy_train1 = np.load("/home/nick/CL14/ImportanceSampling-ex14/metrics_unif-SGD_cifar10_ricap_b05s2/accuracy_per_epoch_train.npy")
Accuracy_train2 = np.load("/home/nick/CL15/ImportanceSampling-ex15/metrics_unif-SGD_cifar10_ricap_b05s2/accuracy_per_epoch_train.npy")
Accuracy_train3 = np.load("/home/nick/CL16/ImportanceSampling-ex16/metrics_unif-SGD_cifar10_ricap_b05s2/accuracy_per_epoch_train.npy")
Accuracy_train4 = np.load("/home/nick/CL17/ImportanceSampling-ex17/metrics_unif-SGD_cifar10_ricap_b05s2/accuracy_per_epoch_train.npy")
# Plot each line with a different color
plt.plot(Accuracy_train1, color='blue', label='Case 1')
plt.plot(Accuracy_train2, color='red', label='Case 2')
plt.plot(Accuracy_train3, color='green', label='Case 3')
plt.plot(Accuracy_train4, color='yellow', label='Case 4')
plt.xlabel('Epoch')
plt.ylabel('Train accuracy')
plt.title('Train accuracy per Epoch under CIFAR-10 & budget=0.5')
# Show the legend with the labels
plt.legend()
# Optionally display the plot
plt.show()
# Save the plot to a file (e.g., PNG)
plt.savefig("/home/nick/PLOTFORRICAP/resultsforacctrain_c10b5.png")

# Reset the figure to create a new independent graph for test accuracy
plt.figure()

#ACC_TEST
Accuracy_test1 = np.load("/home/nick/CL14/ImportanceSampling-ex14/metrics_unif-SGD_cifar10_ricap_b05s2/accuracy_per_epoch_val.npy")
Accuracy_test2 = np.load("/home/nick/CL15/ImportanceSampling-ex15/metrics_unif-SGD_cifar10_ricap_b05s2/accuracy_per_epoch_val.npy")
Accuracy_test3 = np.load("/home/nick/CL16/ImportanceSampling-ex16/metrics_unif-SGD_cifar10_ricap_b05s2/accuracy_per_epoch_val.npy")
Accuracy_test4 = np.load("/home/nick/CL17/ImportanceSampling-ex17/metrics_unif-SGD_cifar10_ricap_b05s2/accuracy_per_epoch_val.npy")
# Plot each line with a different color
plt.plot(Accuracy_test1, color='blue', label='Case 1')
plt.plot(Accuracy_test2, color='red', label='Case 2')
plt.plot(Accuracy_test3, color='green', label='Case 3')
plt.plot(Accuracy_test4, color='yellow', label='Case 4')
plt.xlabel('Epoch')
plt.ylabel('Test accuracy')
plt.title('Test accuracy per Epoch under CIFAR-10 & budget=0.5')
# Show the legend with the labels
plt.legend()
# Optionally display the plot
plt.show()
# Save the plot to a file (e.g., PNG)
plt.savefig("/home/nick/PLOTFORRICAP/resultsforacctest_c10b5.png")

# Reset the figure to create a new independent graph for test accuracy
plt.figure()

#LOSS_TRAIN
Loss_train1 = np.load("/home/nick/CL14/ImportanceSampling-ex14/metrics_unif-SGD_cifar10_ricap_b05s2/LOSS_epoch_train.npy")
Loss_train2 = np.load("/home/nick/CL15/ImportanceSampling-ex15/metrics_unif-SGD_cifar10_ricap_b05s2/LOSS_epoch_train.npy")
Loss_train3 = np.load("/home/nick/CL16/ImportanceSampling-ex16/metrics_unif-SGD_cifar10_ricap_b05s2/LOSS_epoch_train.npy")
Loss_train4 = np.load("/home/nick/CL17/ImportanceSampling-ex17/metrics_unif-SGD_cifar10_ricap_b05s2/LOSS_epoch_train.npy")
# Plot each line with a different color
plt.plot(Loss_train1, color='blue', label='Case 1')
plt.plot(Loss_train2, color='red', label='Case 2')
plt.plot(Loss_train3, color='green', label='Case 3')
plt.plot(Loss_train4, color='yellow', label='Case 4')
plt.xlabel('Epoch')
plt.ylabel('Train loss')
plt.title('Train loss per Epoch under CIFAR-10 & budget=0.5')
# Show the legend with the labels
plt.legend()
# Optionally display the plot
plt.show()
# Save the plot to a file (e.g., PNG)
plt.savefig("/home/nick/PLOTFORRICAP/resultsforlosstrain_c10b5.png")

# Reset the figure to create a new independent graph for test accuracy
plt.figure()

#LOSS_TEST
Loss_test1 = np.load("/home/nick/CL14/ImportanceSampling-ex14/metrics_unif-SGD_cifar10_ricap_b05s2/LOSS_epoch_val.npy")
Loss_test2 = np.load("/home/nick/CL15/ImportanceSampling-ex15/metrics_unif-SGD_cifar10_ricap_b05s2/LOSS_epoch_val.npy")
Loss_test3 = np.load("/home/nick/CL16/ImportanceSampling-ex16/metrics_unif-SGD_cifar10_ricap_b05s2/LOSS_epoch_val.npy")
Loss_test4 = np.load("/home/nick/CL17/ImportanceSampling-ex17/metrics_unif-SGD_cifar10_ricap_b05s2/LOSS_epoch_val.npy")
# Plot each line with a different color
plt.plot(Loss_test1, color='blue', label='Case 1')
plt.plot(Loss_test2, color='red', label='Case 2')
plt.plot(Loss_test3, color='green', label='Case 3')
plt.plot(Loss_test4, color='yellow', label='Case 4')
plt.xlabel('Epoch')
plt.ylabel('Test loss')
plt.title('Test loss per Epoch under CIFAR-10 & budget=0.5')
# Show the legend with the labels
plt.legend()
# Optionally display the plot
plt.show()
# Save the plot to a file (e.g., PNG)
plt.savefig("/home/nick/PLOTFORRICAP/resultsforlosstest_c10b5.png")

# Reset the figure to create a new independent graph for test accuracy
plt.figure()


#############################
######CIFAR100BUDGET0.2######
#############################

#ACC_TRAIN
Accuracy_train1 = np.load("/home/nick/CL14/ImportanceSampling-ex14/metrics_unif-SGD_cifar100_ricap_b02s2/accuracy_per_epoch_train.npy")
Accuracy_train2 = np.load("/home/nick/CL15/ImportanceSampling-ex15/metrics_unif-SGD_cifar100_ricap_b02s2/accuracy_per_epoch_train.npy")
Accuracy_train3 = np.load("/home/nick/CL16/ImportanceSampling-ex16/metrics_unif-SGD_cifar100_ricap_b02s2/accuracy_per_epoch_train.npy")
Accuracy_train4 = np.load("/home/nick/CL17/ImportanceSampling-ex17/metrics_unif-SGD_cifar100_ricap_b02s2/accuracy_per_epoch_train.npy")
# Plot each line with a different color
plt.plot(Accuracy_train1, color='blue', label='Case 1')
plt.plot(Accuracy_train2, color='red', label='Case 2')
plt.plot(Accuracy_train3, color='green', label='Case 3')
plt.plot(Accuracy_train4, color='yellow', label='Case 4')
plt.xlabel('Epoch')
plt.ylabel('Train accuracy')
plt.title('Train accuracy per Epoch under CIFAR-100 & budget=0.2')
# Show the legend with the labels
plt.legend()
# Optionally display the plot
plt.show()
# Save the plot to a file (e.g., PNG)
plt.savefig("/home/nick/PLOTFORRICAP/resultsforacctrain_c100b2.png")

# Reset the figure to create a new independent graph for test accuracy
plt.figure()

#ACC_TEST
Accuracy_test1 = np.load("/home/nick/CL14/ImportanceSampling-ex14/metrics_unif-SGD_cifar100_ricap_b02s2/accuracy_per_epoch_val.npy")
Accuracy_test2 = np.load("/home/nick/CL15/ImportanceSampling-ex15/metrics_unif-SGD_cifar100_ricap_b02s2/accuracy_per_epoch_val.npy")
Accuracy_test3 = np.load("/home/nick/CL16/ImportanceSampling-ex16/metrics_unif-SGD_cifar100_ricap_b02s2/accuracy_per_epoch_val.npy")
Accuracy_test4 = np.load("/home/nick/CL17/ImportanceSampling-ex17/metrics_unif-SGD_cifar100_ricap_b02s2/accuracy_per_epoch_val.npy")
# Plot each line with a different color
plt.plot(Accuracy_test1, color='blue', label='Case 1')
plt.plot(Accuracy_test2, color='red', label='Case 2')
plt.plot(Accuracy_test3, color='green', label='Case 3')
plt.plot(Accuracy_test4, color='yellow', label='Case 4')
plt.xlabel('Epoch')
plt.ylabel('Test accuracy')
plt.title('Test accuracy per Epoch under CIFAR-100 & budget=0.2')
# Show the legend with the labels
plt.legend()
# Optionally display the plot
plt.show()
# Save the plot to a file (e.g., PNG)
plt.savefig("/home/nick/PLOTFORRICAP/resultsforacctest_c100b2.png")

# Reset the figure to create a new independent graph for test accuracy
plt.figure()

#LOSS_TRAIN
Loss_train1 = np.load("/home/nick/CL14/ImportanceSampling-ex14/metrics_unif-SGD_cifar100_ricap_b02s2/LOSS_epoch_train.npy")
Loss_train2 = np.load("/home/nick/CL15/ImportanceSampling-ex15/metrics_unif-SGD_cifar100_ricap_b02s2/LOSS_epoch_train.npy")
Loss_train3 = np.load("/home/nick/CL16/ImportanceSampling-ex16/metrics_unif-SGD_cifar100_ricap_b02s2/LOSS_epoch_train.npy")
Loss_train4 = np.load("/home/nick/CL17/ImportanceSampling-ex17/metrics_unif-SGD_cifar100_ricap_b02s2/LOSS_epoch_train.npy")
# Plot each line with a different color
plt.plot(Loss_train1, color='blue', label='Case 1')
plt.plot(Loss_train2, color='red', label='Case 2')
plt.plot(Loss_train3, color='green', label='Case 3')
plt.plot(Loss_train4, color='yellow', label='Case 4')
plt.xlabel('Epoch')
plt.ylabel('Train loss')
plt.title('Train loss per Epoch under CIFAR-100 & budget=0.2')
# Show the legend with the labels
plt.legend()
# Optionally display the plot
plt.show()
# Save the plot to a file (e.g., PNG)
plt.savefig("/home/nick/PLOTFORRICAP/resultsforlosstrain_c100b2.png")

# Reset the figure to create a new independent graph for test accuracy
plt.figure()

#LOSS_TEST
Loss_test1 = np.load("/home/nick/CL14/ImportanceSampling-ex14/metrics_unif-SGD_cifar100_ricap_b02s2/LOSS_epoch_val.npy")
Loss_test2 = np.load("/home/nick/CL15/ImportanceSampling-ex15/metrics_unif-SGD_cifar100_ricap_b02s2/LOSS_epoch_val.npy")
Loss_test3 = np.load("/home/nick/CL16/ImportanceSampling-ex16/metrics_unif-SGD_cifar100_ricap_b02s2/LOSS_epoch_val.npy")
Loss_test4 = np.load("/home/nick/CL17/ImportanceSampling-ex17/metrics_unif-SGD_cifar100_ricap_b02s2/LOSS_epoch_val.npy")
# Plot each line with a different color
plt.plot(Loss_test1, color='blue', label='Case 1')
plt.plot(Loss_test2, color='red', label='Case 2')
plt.plot(Loss_test3, color='green', label='Case 3')
plt.plot(Loss_test4, color='yellow', label='Case 4')
plt.xlabel('Epoch')
plt.ylabel('Test loss')
plt.title('Test loss per Epoch under CIFAR-100 & budget=0.2')
# Show the legend with the labels
plt.legend()
# Optionally display the plot
plt.show()
# Save the plot to a file (e.g., PNG)
plt.savefig("/home/nick/PLOTFORRICAP/resultsforlosstest_c100b2.png")

# Reset the figure to create a new independent graph for test accuracy
plt.figure()


#############################
######CIFAR100BUDGET0.3######
#############################

#ACC_TRAIN
Accuracy_train1 = np.load("/home/nick/CL14/ImportanceSampling-ex14/metrics_unif-SGD_cifar100_ricap_b03s2/accuracy_per_epoch_train.npy")
Accuracy_train2 = np.load("/home/nick/CL15/ImportanceSampling-ex15/metrics_unif-SGD_cifar100_ricap_b03s2/accuracy_per_epoch_train.npy")
Accuracy_train3 = np.load("/home/nick/CL16/ImportanceSampling-ex16/metrics_unif-SGD_cifar100_ricap_b03s2/accuracy_per_epoch_train.npy")
Accuracy_train4 = np.load("/home/nick/CL17/ImportanceSampling-ex17/metrics_unif-SGD_cifar100_ricap_b03s2/accuracy_per_epoch_train.npy")
# Plot each line with a different color
plt.plot(Accuracy_train1, color='blue', label='Case 1')
plt.plot(Accuracy_train2, color='red', label='Case 2')
plt.plot(Accuracy_train3, color='green', label='Case 3')
plt.plot(Accuracy_train4, color='yellow', label='Case 4')
plt.xlabel('Epoch')
plt.ylabel('Train accuracy')
plt.title('Train accuracy per Epoch under CIFAR-100 & budget=0.3')
# Show the legend with the labels
plt.legend()
# Optionally display the plot
plt.show()
# Save the plot to a file (e.g., PNG)
plt.savefig("/home/nick/PLOTFORRICAP/resultsforacctrain_c100b3.png")

# Reset the figure to create a new independent graph for test accuracy
plt.figure()

#ACC_TEST
Accuracy_test1 = np.load("/home/nick/CL14/ImportanceSampling-ex14/metrics_unif-SGD_cifar100_ricap_b03s2/accuracy_per_epoch_val.npy")
Accuracy_test2 = np.load("/home/nick/CL15/ImportanceSampling-ex15/metrics_unif-SGD_cifar100_ricap_b03s2/accuracy_per_epoch_val.npy")
Accuracy_test3 = np.load("/home/nick/CL16/ImportanceSampling-ex16/metrics_unif-SGD_cifar100_ricap_b03s2/accuracy_per_epoch_val.npy")
Accuracy_test4 = np.load("/home/nick/CL17/ImportanceSampling-ex17/metrics_unif-SGD_cifar100_ricap_b03s2/accuracy_per_epoch_val.npy")
# Plot each line with a different color
plt.plot(Accuracy_test1, color='blue', label='Case 1')
plt.plot(Accuracy_test2, color='red', label='Case 2')
plt.plot(Accuracy_test3, color='green', label='Case 3')
plt.plot(Accuracy_test4, color='yellow', label='Case 4')
plt.xlabel('Epoch')
plt.ylabel('Test accuracy')
plt.title('Test accuracy per Epoch under CIFAR-100 & budget=0.3')
# Show the legend with the labels
plt.legend()
# Optionally display the plot
plt.show()
# Save the plot to a file (e.g., PNG)
plt.savefig("/home/nick/PLOTFORRICAP/resultsforacctest_c100b3.png")

# Reset the figure to create a new independent graph for test accuracy
plt.figure()

#LOSS_TRAIN
Loss_train1 = np.load("/home/nick/CL14/ImportanceSampling-ex14/metrics_unif-SGD_cifar100_ricap_b03s2/LOSS_epoch_train.npy")
Loss_train2 = np.load("/home/nick/CL15/ImportanceSampling-ex15/metrics_unif-SGD_cifar100_ricap_b03s2/LOSS_epoch_train.npy")
Loss_train3 = np.load("/home/nick/CL16/ImportanceSampling-ex16/metrics_unif-SGD_cifar100_ricap_b03s2/LOSS_epoch_train.npy")
Loss_train4 = np.load("/home/nick/CL17/ImportanceSampling-ex17/metrics_unif-SGD_cifar100_ricap_b03s2/LOSS_epoch_train.npy")
# Plot each line with a different color
plt.plot(Loss_train1, color='blue', label='Case 1')
plt.plot(Loss_train2, color='red', label='Case 2')
plt.plot(Loss_train3, color='green', label='Case 3')
plt.plot(Loss_train4, color='yellow', label='Case 4')
plt.xlabel('Epoch')
plt.ylabel('Train loss')
plt.title('Train loss per Epoch under CIFAR-100 & budget=0.3')
# Show the legend with the labels
plt.legend()
# Optionally display the plot
plt.show()
# Save the plot to a file (e.g., PNG)
plt.savefig("/home/nick/PLOTFORRICAP/resultsforlosstrain_c100b3.png")

# Reset the figure to create a new independent graph for test accuracy
plt.figure()

#LOSS_TEST
Loss_test1 = np.load("/home/nick/CL14/ImportanceSampling-ex14/metrics_unif-SGD_cifar100_ricap_b03s2/LOSS_epoch_val.npy")
Loss_test2 = np.load("/home/nick/CL15/ImportanceSampling-ex15/metrics_unif-SGD_cifar100_ricap_b03s2/LOSS_epoch_val.npy")
Loss_test3 = np.load("/home/nick/CL16/ImportanceSampling-ex16/metrics_unif-SGD_cifar100_ricap_b03s2/LOSS_epoch_val.npy")
Loss_test4 = np.load("/home/nick/CL17/ImportanceSampling-ex17/metrics_unif-SGD_cifar100_ricap_b03s2/LOSS_epoch_val.npy")
# Plot each line with a different color
plt.plot(Loss_test1, color='blue', label='Case 1')
plt.plot(Loss_test2, color='red', label='Case 2')
plt.plot(Loss_test3, color='green', label='Case 3')
plt.plot(Loss_test4, color='yellow', label='Case 4')
plt.xlabel('Epoch')
plt.ylabel('Test loss')
plt.title('Test loss per Epoch under CIFAR-100 & budget=0.3')
# Show the legend with the labels
plt.legend()
# Optionally display the plot
plt.show()
# Save the plot to a file (e.g., PNG)
plt.savefig("/home/nick/PLOTFORRICAP/resultsforlosstest_c100b3.png")

# Reset the figure to create a new independent graph for test accuracy
plt.figure()


#############################
######CIFAR100BUDGET0.5######
#############################

#ACC_TRAIN
Accuracy_train1 = np.load("/home/nick/CL14/ImportanceSampling-ex14/metrics_unif-SGD_cifar100_ricap_b05s2/accuracy_per_epoch_train.npy")
Accuracy_train2 = np.load("/home/nick/CL15/ImportanceSampling-ex15/metrics_unif-SGD_cifar100_ricap_b05s2/accuracy_per_epoch_train.npy")
Accuracy_train3 = np.load("/home/nick/CL16/ImportanceSampling-ex16/metrics_unif-SGD_cifar100_ricap_b05s2/accuracy_per_epoch_train.npy")
Accuracy_train4 = np.load("/home/nick/CL17/ImportanceSampling-ex17/metrics_unif-SGD_cifar100_ricap_b05s2/accuracy_per_epoch_train.npy")
# Plot each line with a different color
plt.plot(Accuracy_train1, color='blue', label='Case 1')
plt.plot(Accuracy_train2, color='red', label='Case 2')
plt.plot(Accuracy_train3, color='green', label='Case 3')
plt.plot(Accuracy_train4, color='yellow', label='Case 4')
plt.xlabel('Epoch')
plt.ylabel('Train accuracy')
plt.title('Train accuracy per Epoch under CIFAR-100 & budget=0.5')
# Show the legend with the labels
plt.legend()
# Optionally display the plot
plt.show()
# Save the plot to a file (e.g., PNG)
plt.savefig("/home/nick/PLOTFORRICAP/resultsforacctrain_c100b5.png")

# Reset the figure to create a new independent graph for test accuracy
plt.figure()

#ACC_TEST
Accuracy_test1 = np.load("/home/nick/CL14/ImportanceSampling-ex14/metrics_unif-SGD_cifar100_ricap_b05s2/accuracy_per_epoch_val.npy")
Accuracy_test2 = np.load("/home/nick/CL15/ImportanceSampling-ex15/metrics_unif-SGD_cifar100_ricap_b05s2/accuracy_per_epoch_val.npy")
Accuracy_test3 = np.load("/home/nick/CL16/ImportanceSampling-ex16/metrics_unif-SGD_cifar100_ricap_b05s2/accuracy_per_epoch_val.npy")
Accuracy_test4 = np.load("/home/nick/CL17/ImportanceSampling-ex17/metrics_unif-SGD_cifar100_ricap_b05s2/accuracy_per_epoch_val.npy")
# Plot each line with a different color
plt.plot(Accuracy_test1, color='blue', label='Case 1')
plt.plot(Accuracy_test2, color='red', label='Case 2')
plt.plot(Accuracy_test3, color='green', label='Case 3')
plt.plot(Accuracy_test4, color='yellow', label='Case 4')
plt.xlabel('Epoch')
plt.ylabel('Test accuracy')
plt.title('Test accuracy per Epoch under CIFAR-100 & budget=0.5')
# Show the legend with the labels
plt.legend()
# Optionally display the plot
plt.show()
# Save the plot to a file (e.g., PNG)
plt.savefig("/home/nick/PLOTFORRICAP/resultsforacctest_c100b5.png")

# Reset the figure to create a new independent graph for test accuracy
plt.figure()

#LOSS_TRAIN
Loss_train1 = np.load("/home/nick/CL14/ImportanceSampling-ex14/metrics_unif-SGD_cifar100_ricap_b05s2/LOSS_epoch_train.npy")
Loss_train2 = np.load("/home/nick/CL15/ImportanceSampling-ex15/metrics_unif-SGD_cifar100_ricap_b05s2/LOSS_epoch_train.npy")
Loss_train3 = np.load("/home/nick/CL16/ImportanceSampling-ex16/metrics_unif-SGD_cifar100_ricap_b05s2/LOSS_epoch_train.npy")
Loss_train4 = np.load("/home/nick/CL17/ImportanceSampling-ex17/metrics_unif-SGD_cifar100_ricap_b05s2/LOSS_epoch_train.npy")
# Plot each line with a different color
plt.plot(Loss_train1, color='blue', label='Case 1')
plt.plot(Loss_train2, color='red', label='Case 2')
plt.plot(Loss_train3, color='green', label='Case 3')
plt.plot(Loss_train4, color='yellow', label='Case 4')
plt.xlabel('Epoch')
plt.ylabel('Train loss')
plt.title('Train loss per Epoch under CIFAR-100 & budget=0.5')
# Show the legend with the labels
plt.legend()
# Optionally display the plot
plt.show()
# Save the plot to a file (e.g., PNG)
plt.savefig("/home/nick/PLOTFORRICAP/resultsforlosstrain_c100b5.png")

# Reset the figure to create a new independent graph for test accuracy
plt.figure()

#LOSS_TEST
Loss_test1 = np.load("/home/nick/CL14/ImportanceSampling-ex14/metrics_unif-SGD_cifar100_ricap_b05s2/LOSS_epoch_val.npy")
Loss_test2 = np.load("/home/nick/CL15/ImportanceSampling-ex15/metrics_unif-SGD_cifar100_ricap_b05s2/LOSS_epoch_val.npy")
Loss_test3 = np.load("/home/nick/CL16/ImportanceSampling-ex16/metrics_unif-SGD_cifar100_ricap_b05s2/LOSS_epoch_val.npy")
Loss_test4 = np.load("/home/nick/CL17/ImportanceSampling-ex17/metrics_unif-SGD_cifar100_ricap_b05s2/LOSS_epoch_val.npy")
# Plot each line with a different color
plt.plot(Loss_test1, color='blue', label='Case 1')
plt.plot(Loss_test2, color='red', label='Case 2')
plt.plot(Loss_test3, color='green', label='Case 3')
plt.plot(Loss_test4, color='yellow', label='Case 4')
plt.xlabel('Epoch')
plt.ylabel('Test loss')
plt.title('Test loss per Epoch under CIFAR-100 & budget=0.5')
# Show the legend with the labels
plt.legend()
# Optionally display the plot
plt.show()
# Save the plot to a file (e.g., PNG)
plt.savefig("/home/nick/PLOTFORRICAP/resultsforlosstest_c100b5.png")

# Reset the figure to create a new independent graph for test accuracy
plt.figure()