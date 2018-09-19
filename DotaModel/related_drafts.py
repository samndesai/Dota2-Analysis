from gensim.models.doc2vec import Doc2Vec
import pickle
model_path = 'dota_model'


with open('dotaname_to_id.pkl','rb') as file:
    nametoid = pickle.load(file)
with open('id_to_dotaname.pkl','rb') as file:
    idtoname = pickle.load(file)
with open('match_picks.pkl','rb') as file:
    match_picks = pickle.load(file)

for i in range(11):
    print(idtoname[i],nametoid[idtoname[i]])

model = Doc2Vec.load(model_path)


def related_matches(heroes):
    print(len(heroes))
    inputlist = []
    for hero in heroes:
        inputlist.append(str(nametoid[hero]))
    print("Matches similar to:", heroes,len(heroes))
    try:
        for terms in model.docvecs.most_similar(positive=[model.infer_vector(inputlist)], topn=15):
            print("\t", end='')
            for heroes in match_picks[int(terms[0])].rstrip().lstrip().split(" "):
                print(idtoname[int(heroes)], end=', ')
            print(terms[1])
    except KeyError:
        print("KeyError")
        pass
    print()

while True:
    related_matches(input("Enter heroes to query (separated by space):").rstrip().lstrip().split(","))