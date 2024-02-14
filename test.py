from sklearn.preprocessing import MultiLabelBinarizer
labels_names = ['Mass', 'Consolidation', 'Atelectasis', 'Hernia', 'Cardiomegaly', 'Pleural_Thickening', 'Pneumonia', 'Nodule', 'Emphysema', 'Infiltration', 'Edema', 'No Finding', 'Effusion', 'Pneumothorax', 'Fibrosis']
mlb = MultiLabelBinarizer()
mlb.fit([labels_names])
a = mlb.transform([['Consolidation']])[0]
print(a)

