# Encode GamePopularity: Low=0, Medium=1, High=2
label_map = {'Low': 0, 'Medium': 1, 'High': 2}
df['GamePopularity_enc'] = df['GamePopularity'].map(label_map)

print('Class distribution:')
print(df['GamePopularity'].value_counts())

X = df.drop(columns=['GamePopularity', 'GamePopularity_enc'])
y = df['GamePopularity_enc']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
# Save label_map
with open('label_map_cls.pkl', 'wb') as f:
    pickle.dump(label_map, f)
print('Saved label_map_cls.pkl')


K_FEATURES = 20   # keep top 20 features

selector = SelectKBest(score_func=f_classif, k=K_FEATURES)
selector.fit(X_train, y_train)

selected_mask = selector.get_support()
selected_features = X_train.columns[selected_mask].tolist()

print(f'Selected {len(selected_features)} features:')
print(selected_features)

X_train_sel = X_train[selected_features]
X_test_sel  = X_test[selected_features]

# Save selector
with open('selector_cls.pkl', 'wb') as f:
    pickle.dump(selector, f)
print('Saved selector_cls.pkl')


# Scaling for knn
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train_sel)
X_test_sc  = scaler.transform(X_test_sel)

# Save scaler
with open('scaler_cls.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print('Saved scaler_cls.pkl')

## Hyper parameter tunning

# k neighbours
# k is changing , weight is constant ( uniform )
k_values = [3, 5, 7, 9, 11]
k_results = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k, weights='uniform', metric='minkowski', n_jobs=-1)
    knn.fit(X_train_sc, y_train)
    acc = accuracy_score(y_test, knn.predict(X_test_sc))
    k_results.append({'k': k, 'accuracy': acc})
    print(f'  k={k:2d} → Accuracy = {acc:.4f}')

k_df = pd.DataFrame(k_results)


# visualizing
plt.figure(figsize=(7, 4))
plt.plot(k_df['k'], k_df['accuracy'], marker='o', color='steelblue', linewidth=2)
plt.xticks(k_values)
plt.xlabel('n_neighbors (k)')
plt.ylabel('Test Accuracy')
plt.title('KNN – Effect of n_neighbors (weights=uniform)')
plt.grid(True)
plt.tight_layout()
plt.show()


best_k = k_df.loc[k_df['accuracy'].idxmax(), 'k']
print(f'\nBest k = {best_k}')


## claude did this
def inv_sq(distances):
    """Custom weight: inverse-square of distance."""
    return 1 / (distances ** 2 + 1e-10)

weight_options = ['uniform', 'distance', inv_sq]
weight_labels  = ['uniform', 'distance', 'inv_sq']
w_results = []
# elmafrod it will choose best k value as a constant , then change the weight
for w, label in zip(weight_options, weight_labels):
    knn = KNeighborsClassifier(n_neighbors=int(best_k), weights=w, metric='minkowski', n_jobs=-1)
    knn.fit(X_train_sc, y_train)
    acc = accuracy_score(y_test, knn.predict(X_test_sc))
    w_results.append({'weights': label, 'accuracy': acc})
    print(f'  weights={label:12s} → Accuracy = {acc:.4f}')

w_df = pd.DataFrame(w_results)

plt.figure(figsize=(6, 4))
plt.bar(w_df['weights'], w_df['accuracy'], color=['steelblue', 'darkorange', 'green'])
plt.xlabel('Weights')
plt.ylabel('Test Accuracy')
plt.title(f'KNN – Effect of weights (k={int(best_k)})')
plt.ylim(0, 1)
for i, v in enumerate(w_df['accuracy']):
    plt.text(i, v + 0.005, f'{v:.4f}', ha='center', fontsize=9)
plt.tight_layout()
plt.show()


best_w_label = w_df.loc[w_df['accuracy'].idxmax(), 'weights']
best_w_func  = weight_options[weight_labels.index(best_w_label)]
print(f'\nBest weights = {best_w_label}')
