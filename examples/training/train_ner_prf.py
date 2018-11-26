import plac
import json
import random
import logging
import spacy
from spacy.util import minibatch, compounding
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from spacy.gold import GoldParse, biluo_tags_from_offsets
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


def make_doc(nlp, input_):
    if isinstance(input_, bytes):
        input_ = input_.decode('utf8')
    if isinstance(input_, str):
        return nlp.tokenizer(input_)
    else:
        return spacy.Doc(nlp.vocab, words=input_)


def make_gold(nlp, input_, annotations):
    doc = make_doc(nlp, input_)
    #print(annotations)
    annot_tuples = (None, None, None, None, None, annotations)
    gold = spacy.gold.GoldParse.from_annot_tuples(doc, annot_tuples)
    return gold


def evaluate(nlp, examples):
    scorer = spacy.scorer.Scorer()
    for sent in examples:
        if 'sentences' in sent:
            for text, annot in sent['sentences']:
                gold = make_gold(nlp, text, annot.get('entities'))
                doc = nlp(text)
                scorer.score(doc, gold, verbose=False)
    return scorer.scores


def report_scores(i, loss, scores):
    #precision = '%.2f' % scores['ents_p']
    #recall = '%.2f' % scores['ents_r']
    #f_measure = '%.2f' % scores['ents_f']
    precision = '%f' % scores['ents_p']
    recall = '%f' % scores['ents_r']
    f_measure = '%f' % scores['ents_f']
    print('%d %s %s %s %s' % (i, float(loss), precision, recall, f_measure))


def save_model(dirname, nlp, scores):
    if dirname is not None:
        dirname = Path(dirname)
        if not dirname.exists():
            dirname.mkdir()
        nlp.to_disk(dirname)
        # update scores
        with open(dirname / 'accuracy.json', "w") as acc_r:
            obj = {}
            obj['ents_p'] = scores['ents_p']
            obj['ents_r'] = scores['ents_r']
            obj['ents_f'] = scores['ents_f']
            acc_r.write(json.dumps(obj, indent=4))
        with open(dirname / 'meta.json', "r+") as acc_r:
            obj = json.loads(acc_r.read())
            obj['accuracy']['ents_p'] = scores['ents_p']
            obj['accuracy']['ents_r'] = scores['ents_r']
            obj['accuracy']['ents_f'] = scores['ents_f']
            acc_r.seek(0)
            acc_r.write(json.dumps(obj, indent=4))
            acc_r.truncate()


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
    
    with open(eval_file, "r") as devfile_:
        TEST_DATA = hinted_tuple_hook(json.load(devfile_))

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    print("#Epoch", "Loss", "P", "R", "F")
    try:
        with nlp.disable_pipes(*other_pipes):  # only train NER
            #if model is None:
            #    optimizer = nlp.begin_training()
            #else:
                # Note that 'begin_training' initializes the models, so it'll zero out
                # existing entity types.
            #    optimizer = nlp.entity.create_optimizer()

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
                #print('Losses', losses)
                scores = evaluate(nlp, TEST_DATA)
                report_scores(itn, losses['ner'], scores)
                # TODO: save every iteration in dirs
                save_model(output_dir / ("model%s" % str(itn)), nlp, scores)

    finally:
        with nlp.use_params(optimizer.averages):
            scores = evaluate(nlp, TEST_DATA)
            report_scores(itn+1, losses['ner'], scores)
            save_model(output_dir / "model-final", nlp, scores)
            print("Saved models to: ", output_dir)

    """
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

        #f=open("resume"+str(c)+".txt","w")
        doc_to_test=nlp(text)
        d={}
        for ent in doc_to_test.ents:
            d[ent.label_]=[]
        for ent in doc_to_test.ents:
            d[ent.label_].append(ent.text)

        #for i in set(d.keys()):
            #f.write("\n\n")
            #f.write(i +":"+"\n")
            #for j in set(d[i]):
            #    f.write(j.replace('\n','')+"\n")

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
    """


if __name__ == '__main__':
    plac.call(main)
