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

def related_heroes(hero_name):
    print("Showing heroes similar to:", hero_name, nametoid[hero_name])
    try:
        for terms in model.most_similar(str(nametoid[hero_name]),topn= 25):
            print("\t", idtoname[int(terms[0])], terms[1])
        related_match_picks(hero_name)
    except KeyError:
        print("KeyError")
        pass
    print()
def related_match_picks(hero_name):
    print("Matches similar to:", hero_name, nametoid[hero_name])
    try:
        for terms in model.docvecs.most_similar(positive=[model.infer_vector(str(nametoid[hero_name]))],topn= 15):
            print("\t",end='')
            for heroes in match_picks[int(terms[0])].rstrip().lstrip().split(" "):
                print(idtoname[int(heroes)], end=', ')
            print(terms[1])
    except KeyError:
        print("KeyError")
        pass
    print()

while True:
    related_heroes(input("Enter exact hero name: "))