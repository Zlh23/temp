import torch
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


class PCA:
    def __init__(self, output_dim) -> None:
        self.output_dim = output_dim

    def fit(self, X_data):
        N = len(X_data)
        H = torch.eye(n=N) - 1 / N * (torch.matmul(torch.ones(size=(N, 1)), torch.ones(size=(1, N))))
        X_data = torch.matmul(H, X_data)
        _, _, v = torch.svd(X_data)
        self.base = v[:, :self.output_dim]

    def fit_transform(self, X_data):
        self.fit(X_data)
        return self.transform(X_data)

    def transform(self, X_data):
        return torch.matmul(X_data, self.base)

    def inverse_transform(self, X_data):
        return torch.matmul(X_data, self.base.T)

digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits.data,digits.target, test_size=0.2, random_state=42)

print(X_train[0])
X_train = torch.tensor(X_train,dtype=torch.float)
X_test = torch.tensor(X_test,dtype=torch.float)
y_train = torch.tensor(y_train,dtype=torch.float)
y_test = torch.tensor(y_test,dtype=torch.float)

pca = PCA(2)
print(X_train[0])
X_train_pca = pca.fit_transform(X_train)
plt.scatter(X_train_pca[:,0],X_train_pca[:,1],c=y_train)
print(X_train_pca[:,0])
print(X_train_pca[:,1])
print(y_train)

plt.figure()
plt.subplot(331)

for i,dim in enumerate([2,10,20,30,40,50,60]):
    pca = PCA(dim)
    X_train_pca = pca.fit_transform(X_train)
    X_data = pca.inverse_transform(X_data=X_train_pca)
    plt.subplot(2,4,i+1)
    plt.imshow(X_data[0].view(8,8))
plt.subplot(2,4,8)
plt.imshow(X_train[0].view(8,8))
plt.show()