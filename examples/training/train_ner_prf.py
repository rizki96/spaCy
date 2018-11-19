import plac
import json
import random
import logging
import spacy
from spacy.util import minibatch, compounding
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from spacy.gold import GoldParse
from spacy.scorer import Scorer
from sklearn.metrics import accuracy_score
from pathlib import Path


def hinted_tuple_hook(obj):
    if isinstance(obj, dict) and '__tuple__' in obj:
        return tuple([hinted_tuple_hook(o) for o in obj['items']])
    elif isinstance(obj,list):
        return [hinted_tuple_hook(o) for o in obj]
    elif isinstance(obj, dict):
        return {key: hinted_tuple_hook(value) for key, value in obj.items()}
    else:
        return obj


@plac.annotations(
    train_file=("Training file for ner model", "option", "t", str),
    eval_file=("Evaluation data file to test ner model", "option", "e", str),
    model=("Model name. Defaults to blank 'en' model.", "option", "m", str),
    output_dir=("Optional output directory", "option", "o", Path),
    n_iter=("Number of training iterations", "option", "n", int),
    dropout=("Number of dropout parameter", "option", "d", float))
def main(train_file=None, eval_file=None, model=None, output_dir=None, n_iter=10, dropout=0.5):
    """Load the model, set up the pipeline and train the entity recognizer."""
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank('en')  # create blank Language class
        print("Created blank 'en' model")

    # create the built-in pipeline components and add them to the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner, last=True)
    # otherwise, get it so we can add labels
    else:
        ner = nlp.get_pipe('ner')

    with open(train_file, "r") as trainfile_:
        TRAIN_DATA = hinted_tuple_hook(json.load(trainfile_))

    # add labels
    for sent in TRAIN_DATA:
        if 'sentences' in sent:
            for _, annotations in sent['sentences']:
                for ent in annotations.get('entities'):
                    ner.add_label(ent[2])

    train_sents = []
    for sent in TRAIN_DATA:
        if 'sentences' in sent:
            train_sents.extend(sent['sentences'])
    
    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):  # only train NER
        #print("#Epoch", "Loss", "P", "R", "F")
        optimizer = nlp.begin_training()
        for itn in range(n_iter):
            random.shuffle(train_sents)
            losses = {}
            # batch up the examples using spaCy's minibatch
            batches = minibatch(train_sents, size=compounding(4., 32., 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(
                    texts,  # batch of texts
                    annotations,  # batch of annotations
                    drop=dropout,  # dropout - make it harder to memorise data
                    sgd=optimizer,  # callable to update weights
                    losses=losses)
            print('Losses', losses)

    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

    with open(eval_file, "r") as devfile_:
        TEST_DATA = hinted_tuple_hook(json.load(devfile_))

    test_sents = []
    for sent in TEST_DATA:
        if 'sentences' in sent:
            test_sents.extend(sent['sentences'])
    
    tp=0
    tr=0
    tf=0

    ta=0
    c=0
    for text,annot in test_sents:

        f=open("resume"+str(c)+".txt","w")
        doc_to_test=nlp(text)
        d={}
        for ent in doc_to_test.ents:
            d[ent.label_]=[]
        for ent in doc_to_test.ents:
            d[ent.label_].append(ent.text)

        for i in set(d.keys()):

            f.write("\n\n")
            f.write(i +":"+"\n")
            for j in set(d[i]):
                f.write(j.replace('\n','')+"\n")
        d={}
        for ent in doc_to_test.ents:
            d[ent.label_]=[0,0,0,0,0,0]
        for ent in doc_to_test.ents:
            doc_gold_text= nlp.make_doc(text)
            gold = GoldParse(doc_gold_text, entities=annot.get("entities"))
            y_true = [ent.label_ if ent.label_ in x else 'Not '+ent.label_ for x in gold.ner]
            y_pred = [x.ent_type_ if x.ent_type_ ==ent.label_ else 'Not '+ent.label_ for x in doc_to_test]  
            if(d[ent.label_][0]==0):
                #f.write("For Entity "+ent.label_+"\n")   
                #f.write(classification_report(y_true, y_pred)+"\n")
                (p,r,f,s)= precision_recall_fscore_support(y_true,y_pred,average='weighted')
                a=accuracy_score(y_true,y_pred)
                d[ent.label_][0]=1
                d[ent.label_][1]+=p
                d[ent.label_][2]+=r
                d[ent.label_][3]+=f
                d[ent.label_][4]+=a
                d[ent.label_][5]+=1
        c+=1
    for i in d:
        print("\n For Entity "+i+"\n")
        print("Accuracy : "+str((d[i][4]/d[i][5])*100)+"%")
        print("Precision : "+str(d[i][1]/d[i][5]))
        print("Recall : "+str(d[i][2]/d[i][5]))
        print("F-score : "+str(d[i][3]/d[i][5]))


if __name__ == '__main__':
    plac.call(main)
