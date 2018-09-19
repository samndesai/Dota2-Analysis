from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedLineDocument
from time import localtime,strftime

# document for training
doc_path = 'dota_picks'
corpus = TaggedLineDocument(doc_path)

# Doc2Vec parameters; self explanatory
vector_size = 50
window_size = 5
min_count = 0
sampling_threshold = 1e-4
negative_size = 5
train_epoch = 100
dm = 0 #0 = dbow; 1 = dmpv
worker_count = 8 #number of parallel processes


model = Doc2Vec(size=vector_size, window=window_size, min_count=min_count, sample=sampling_threshold, workers=worker_count, hs=0, dm=dm, negative=negative_size, dbow_words=1, dm_concat=1)
print("Building Vocab:", strftime("%a, %d %b %Y, %H:%M:%S", localtime()))
model.build_vocab(corpus)
print("Built Vocab:", strftime("%a, %d %b %Y, %H:%M:%S", localtime()))
for epoch in range(10):
    print("Training: Epoch-", epoch, strftime("%a, %d %b %Y, %H:%M:%S", localtime()))
    model.train(corpus)
    model.alpha -= 0.002 # decrease the learning rate
    model.min_alpha = model.alpha # fix the learning rate, no decay
    print("Training 2: Epoch-", epoch, strftime("%a, %d %b %Y, %H:%M:%S", localtime()))
    model.train(corpus)


print("Model Trained!", strftime("%a, %d %b %Y, %H:%M:%S", localtime()))
print()

#save model
save_path = 'dota_model'
model.save(save_path)
print("Model Saved", strftime("%a, %d %b %Y, %H:%M:%S", localtime()))