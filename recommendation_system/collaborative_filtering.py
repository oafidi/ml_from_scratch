import pandas as pd
import numpy as np

collab_filtered_data = {
	'client_id': [5, 6, 5, 6, 7, 8, 7],
	'categorie': ['Électronique', 'Électronique', 'Beauté', 'Beauté', 'Beauté', 'Jeux', 'Jeux'],
	'score': [4, 4, 2, 2, 1, 1, 2]
}

def cosine_similarity(user1, user2):
    return np.dot(user1, user2) / (np.linalg.norm(user2) * np.linalg.norm(user1))

def fn(row, df, users):
    for user in users:
        row[user] = cosine_similarity(df.loc[row.name], df.loc[user])
    return row


def recommendation_system(collab_filtered_data, client_id, k=1):
    df = pd.DataFrame(collab_filtered_data)
    df = df.pivot(index="client_id", columns="categorie", values="score").fillna(value=0).astype(int)
    users = list(df.index)
    cosine_matrix = pd.DataFrame(float(0), index=users, columns=users)
    cosine_matrix = cosine_matrix.apply(fn, args=(df, users), axis=1)
    print("client_id:", client_id)
    print("leurs similaires: ",list(cosine_matrix.loc[client_id].drop(client_id).sort_values(ascending=False)[:k].index))
    return list(cosine_matrix.loc[client_id].drop(client_id).sort_values(ascending=False)[:k].index)

if __name__=="__main__":
    print(recommendation_system(collab_filtered_data, 5, 2))
