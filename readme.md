# Essential Protein Prediction

This repo is about using **Gene Spetrum** data and **Protein Topology** data to predict essential protein.

The code involves two models in deep learning implemented by tensorflow-1.1.0:
* Convolutional Neural Network [Details](https://arxiv.org/abs/1408.5882)
* Node2vec [Details](https://github.com/aditya-grover/node2vec)

---
```python
plt.title('ROC')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
for i, filename in enumerate(image_list):
    with open('plot_cache/' + filename, 'rb') as file:
        y_pred, y_label = pickle.load(file)
        fpr, tpr, thresholds = roc_curve(y_label, y_pred)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, 'k--', color=colors[i], label='%s AUC:%.3f' % (legends[i], roc_auc), lw=2)

plt.legend(loc="lower right", fontsize='medium')  # after plt.plot
plt.savefig('ROC.jpg')
plt.show()

```

```python
plt.title('Precision/Recall Curve')  # give plot a title
plt.xlabel('Recall')  # make axis labels
plt.ylabel('Precision')
for i, filename in enumerate(image_list):
    with open('plot_cache/' + filename, 'rb') as file:
        y_score, y_label = pickle.load(file)
        precision, recall, _ = precision_recall_curve(y_label, y_score)
        aupr = auc(recall, precision)
        plt.plot(recall, precision, 'k--', color=colors[i], label='%s AUPR:%.3f' % (legends[i], aupr), lw=2)

plt.legend(loc="lower right", fontsize='small')  # after plt.plot

plt.savefig('PR.jpg')
plt.show()
```

![ROC](https://github.com/feizhihui/Essential-Protein-Prediction/blob/master/ROC.jpg)

![Precision/Recall Curve](https://github.com/feizhihui/Essential-Protein-Prediction/blob/master/PR.jpg)

