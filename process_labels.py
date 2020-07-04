from imagenetlabels import idx_to_labels

f = open('labels.txt', 'w')
for i in range(1000):
    f.write(idx_to_labels[i].split(',')[0])
    f.write('\n')
f.close()
