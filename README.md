# Επεξεργασία Φυσικής Γλώσσας

## Παραδοτέο 1Α:
**Σκοπός:** να κάνουμε αναδιατύπωση και βασική δίορθωση γραμματικής σε αγγλικές προτάσεις, με χρήση NLP εραγλείων και τεχνικών

**Επεξήγηση:**

Αρχικά, φροτώνουμε τις βασικές βιβλιοθήκες που θα χρησιμοποιήσουμε στο κομμάτι του κώδικα. Τους transformers του χρησιμοποιούμε προκειμενου να φορτώσουμε το T5 μοντέλο παραφράσεων από το HuggingFace. Το spacy και nltk μας βοηθούν για tokenization και part of speech tagging.

```python 
from transformers import AutoModelForSeq2SeqLM,AutoTokenizer # για το paraphrasing
from nltk.tokenize import TreebankWordTokenizer
import spacy
```

-->getTags(text): Χρησιμοποιεί το Spacy προκειμένου να πάρουμε τη λίστα με τη μορφή (λέξη, γραμματικο tag) για κάθε λέξη του κειμένου που περνά ως όρισμα.

```python
def getTags(text):
    wordTag=nlp(text)
    return [(token.text,token.tag_) for token in wordTag]
```

-->correct_grammar(text): Στη συγκεκριμένη συνάρτηση παίρνουμε τη λίστα από το getTags και για κάθε λέξη παίρνουμε access στην ίδια τη λέξη αλλά και το tag τους για να βρούμε τι μέρος του λόγου είναι. Από εκεί εφαρμόζουμε απλούς χειροποίητους κανόνες, οι οποίοι είναι:
+Αν η πρόταση ξεκινάει με ουσιαστικό χώρις άρθρο, προσθέτουμε το 'a'.
+Αν δύο ουσιαστικά εμφανίζονται διαδοχικά, τότε προσθέτουμε ανάμεσά τους ένα of.
+Αν βρεθεί η λέξη at πριν από χρονική λέξη, αντικαθιστάται με το in.

```python
#Κανόνας Πρώτος αν δεν πρόταση ξεκινά με ουσιαστικό χωρίς άρθρο
        if t=="NN" and counter==0:
            correct.append("a")
            correct.append(word)
        #Εφαρμοφή δεύτερου κανόνα αν έχουμε ουσιαστικό που ακολουθεί άλλο ουσιαστικό χωρίς σύνδεση
        elif t=="NN" and tag[counter-1][1] not in ["DT","JJ"]:
            #Αν υπάρχει πριν ουσιαστικό, θέλουμε σύνδεση
            if tag[counter-1][1]=="NN":
                correct.append("of")
                correct.append(word)
            else:
                correct.append(word)
        #Εφαρμογή τρίτου κάνονα αν βρούμε at και έχουμε χρονική λέξη
        elif word.lower()=="at" and counter+1<len(tag):
            nextWord,nextTag=tag[counter+1]
            if nextWord.lower() in ["recent","next","past"]:
                correct.append("in")
            else:
                correct.append(word)
        else:
            #Δεν βρίσκεται κάποιος κανόνας
            correct.append(word)
        counter+=1
```

-->paraphrase(text): Εδώ εφαρμόζεται το T5 μοντέλο για να αναδιατυπώσουμε το κείμενο του ορίσματος. Χρησιμοποιούμε ρητή μορφή για paraphrase. Στο τέλος, παράγουμε ένα κείμενο με τις εξής παραμέτρους:
+no_repeat_ngram_size=2, αποτρέπει την επανάληψη διπλών λέξεων.
+num_beams=5, Beam search με 5 διαδρομές για ποιοτική έξοδο.
+early_stopping=True, σταματά την αναζήτηση όταν βρεθεί επαρκής πρόταση.

```python
def paraphrase(text):
    #Προσθέτουμε την οδηγία->αναδιατύπωση
    input=f"paraphrase:{text}</s>"
    
    #Δημιουργία tokens
    tokens=tokenizer.encode(input,return_tensors="pt")
    
    #Δημιουργία απάντησης
    paraphrase_text=model.generate(
        tokens,
        no_repeat_ngram_size=2,#Αποφευγεί να πεί την ίδια ακολουθία
        num_beams=5,
        early_stopping=True,
    )
    
    return tokenizer.decode(paraphrase_text[0],skip_special_tokens=True)
```

Η έξοδος του μοντέλου μετατρέπεται ξανά σε κανονικό string αφαιρώντας tokens όπως </s>,<pad>.

## Παραδοτέο 1B:
**Ο κώδικας στο συγκεκριμένο ερώτημα:**Αξιοποιεί το pipeline API από τη βιβλιοθήκη transformers της Hugging Face, για να μπορέσουμε να υλοποιήουμε αυτόματη παράφραση αγγλικών προτάσεων.

**Δομή:**Οι κώδικες περιλάμβάνουν (1) τη φόρτωση κατάλληλου μοντέλου μέσω pipeline,(2) ορισμό κειμένου για παράφραση, (3) εκτέλεση της παράφρασης μέσω της συνάρτησης pipeline με ενεργή ρύθμιση για ποικιλία και (4) εκτύπωση του αποτελέσματος.

Μοντέλο Bart:

```python
from transformers import pipeline

#Φορτώνουμε τον pipline paraphraser bart
paraphraserBart=pipeline("text2text-generation",model="facebook/bart-large-cnn")

text="""Today is our dragon boat festival, in our Chinese culture, to celebrate it with all safe and great in
our lives. Hope you too, to enjoy it as my deepest wishes.
Thank your message to show our words to the doctor, as his next contract checking, to all of us.
I got this message to see the approved message. In fact, I have received the message from the
professor, to show me, this, a couple of days ago. I am very appreciated the full support of the
professor, for our Springer proceedings publication"""

paraphraseText=paraphraserBart(text,max_length=256,do_sample=True)#Παραγωγή ποικιλία
print("method Bart:",paraphraseText[0]['generated_text'])
```
Μοντέλο T5:
```python
from transformers import pipeline

#Φόρτωση 
paraphraserT5=pipeline("text2text-generation",model="Vamsi/T5_Paraphrase_Paws")

text="""Today is our dragon boat festival, in our Chinese culture, to celebrate it with all safe and great in
our lives. Hope you too, to enjoy it as my deepest wishes.
Thank your message to show our words to the doctor, as his next contract checking, to all of us.
I got this message to see the approved message. In fact, I have received the message from the
professor, to show me, this, a couple of days ago. I am very appreciated the full support of the
professor, for our Springer proceedings publication"""

#Υπόδειξη task
inputText="paraphrase: "+text
out=paraphraserT5(inputText,max_length=256,do_sample=True)
print("method T5:",out[0]['generated_text'])

```
Μοντέλο Pegasus:


```python

from transformers import pipeline

#Φόρτωση 
paraphraserPegasus=pipeline("text2text-generation",model="google/pegasus-xsum")

text="""Today is our dragon boat festival, in our Chinese culture, to celebrate it with all safe and great in
our lives. Hope you too, to enjoy it as my deepest wishes.
Thank your message to show our words to the doctor, as his next contract checking, to all of us.
I got this message to see the approved message. In fact, I have received the message from the
professor, to show me, this, a couple of days ago. I am very appreciated the full support of the
professor, for our Springer proceedings publication"""

output=paraphraserPegasus(text,max_length=256,do_sample=True)
print("method Pegasus:",out[0]['generated_text'])
```

Και τα τρία έχουν τις παραμέτρους:
+max_length=256, μέγιστο μήκος παραγόμενου κειμένου
+do_sample=True, ενεργοποιεί τυχαιότητα για πιο ποικιλόμορφη και δημιουργική έξοδο.

## Παραδοτέο 1C:
**Σκοπός:**για να εκτιμήσουμε την ποιότητα των παραφράσεων που παράγουν τα μοντέλα Bart, T5, Pegasus, χρησιμοποιούμε το **metric BertScore**, το οποίο αξιολογεί τη σημασιολογική ομοιόητα μτεαξύ της αρχικής και της παραφρασμένης πρότασης.

**BERTScore:** αξιολογεί ένα παραγομένο κείμενο με βάση τις εννοιολογικές σχέσεις που προκύπτουν από τα embedding του προεκπαιδευμένου μοντέλου BERT. Για λόγους ταχύτητας χρησιμοποιούμε το μοντέλο bert-base-uncased και επιστρέφει:
+P,ποσό καλά τα παραγόμενα tokens ταιριάζουν με το αναμενόμενο.
+R, πόσο καλά το αναμενόμενο καλύπτει το παραγόμενο.
+F ο μέσος όρος αυτών των δύο. Αυτό διατήρουμε στον συγκεντροτικό πίνακα αποτελεσμάτων που δημιουργούμε με τη βοήθεια του pandas.

#Θα υπολογίσουμε τα score
P2,R2,F2=score(bartText,originalText,lang="en",verbose=True,model_type="bert-base-uncased")
P3,R3,F3=score(t5Text,originalText,lang="en",verbose=True,model_type="bert-base-uncased")
P4,R4,F4=score(pegasusText,originalText,lang="en",verbose=True,model_type="bert-base-uncased")

```python
#Δημιουργία Πίνακα Αποτελεσμάτων
table=pd.DataFrame(
    {
        "Methods":["Bart","T5","Pegasus"],
        "BERTScore":[F2[0].item(),F3[0].item(),F4[0].item()]
    }
)
print(table)

```
## Παραδοτεό 2:

**Σκοπός:**το παραδοτέο στοχεύει να μετρηθεί η σημασιολογική ομοιότητα μεταξύ αρχικών και παραφρασμένων προτάσεων και να οποτικοποηθεί η εγγύτητα με τη χρήση του PCA.

Με τη βοήθεια του μοντέλου all-MiniLM-L6-v2 που παρέχει η Sentence Transformes, μετατρέπουμε τα κείμενα σε πυκνά διανύσματα.

### Μέρος 1Α
Το συγκεκριμένο μέρος αφορά την ομοιότητα προτάσεων με Cosine Similarity, για κάθε πρόταση παράγονται τα διανύσματα embedding μέσω του SentenceTransformer και υπολογίζεται το συνημίτονο της γωνίας. Με λίγα λόγια, πόσο κόντα βρίσκονται στο νοηματικό χώρο. Χρησιμοποιούμε για ευκολότερη συγκριτική ερμηνεία μόνο το πρώτο δεκαδικό και έχει διάστημα τιμών με [0,1], με 1 να είναι η πλήρης ταύτιση.

```python
#Φόρτωση μόντέλου
model=SentenceTransformer('all-MiniLM-L6-v2')

#Περνάμε τις προτάσεις
text1="Today is our dragon boat festival, in our Chinese culture, to celebrate it with all safe and great in our lives."
paraphrase_text1=" a Today is our dragon of boat of festival , in our Chinese culture , to celebrate it with all safe and great"

text2="Anyway, I believe the team, although bit delay and less communication at recent days, they really tried best for paper and cooperation."
paraphrase_text2=" Anyway , I believe the team , although a bit delayed and less communication in recent days ,"

#Μετατροπή σε διανυσματικές αναπαραστάσεις
emb1=model.encode([text1,paraphrase_text1])
emb2=model.encode([text2,paraphrase_text2])


#Υπολογισμός βαθμολογίας συνημίτονου
similarity1=cosine_similarity([emb1[0],emb1[1]])[0][0]
similarity2=cosine_similarity([emb2[0],emb2[1]])[0][0]
```

### Μέρος 1Β
Το μόνο διαφορετικό κομμάτι είναι ότι δεν εξετάζουμε μεμονωμένες προτάσεις, εξετάζονται ολόκληρα τα κείμενα,δηλαδή το αρχικό μαζί με τις παραφράσεις που δημιουργήθηκαν από το BART, T5 ,Pegasus.

```python

emb_bart=model.encode([originalText,paraphraseTextBart])
emb_T5=model.encode([originalText,paraphraseTextT5])
emb_pegasus=model.encode([originalText,paraphraseTextPegasus])


#Υπολογισμός βαθμολογίας συνημίτονου
similarityBART=cosine_similarity([emb_bart[0],emb_bart[1]])[0][0]
similarityT5=cosine_similarity([emb_T5[0],emb_T5[1]])[0][0]
similarityPegasus=cosine_similarity([emb_pegasus[0],emb_pegasus[1]])[0][0]
```

Ο υψηλότερος βαθμός υποδηλώνει καλύτερη διατήρηση του αρχικου νοήματος.

### Οπτικοποιήση PCA

Κάθε σύνολο κειμένων μετατρέπεται σε 384-διάστατα διανύσματα. Αυτό το κάνουμε προκειμένου να οπτικοποιήσουμε τις θέσεις των αρχικών και παραφρασμένων κειμένων στον χώρο, εμφαρμόζοντας PCA σε 2 διαστάσεις. Το κάθε διάγραμμα δείχνει πόσο κόντα βρίσκονται οι παραφράσεις από το κείμενο.

```python

#Δημιουργία διανυσματικών αναπαραστάσεων
vectorRepre=model.encode(listOftexts) #Κάθε vector έχει 384 διαστάσεις

#Μείωση διαστάσεων μέσω PCA
pca=PCA(n_components=2)
#Μετατροπή
reducedVector=pca.fit_transform(vectorRepre)

#Δημιουργία διαγράμματος πλαισού 8*6
plt.figure(figsize=(8,6))
#προσβάση στους άξονες-->διάγραμμα διασπόρας
plt.scatter(reducedVector[:,0],reducedVector[:,1])

types=["original","bart","T5","Pegasus"]
#Προσθήκη ετικετών
for i, text in enumerate(listOftexts):
    #Τοποθέτηση ονόματος στις συντεταγμένες
    plt.annotate(types[i],(reducedVector[i,0],reducedVector[i,1]))
    
plt.title("PCA text A:")
plt.show()
```

Οι τελικες 2D συντεταγμένες απεικονίζονται με matplotlib σε διάγραμμα διασποράς.

**Κοντινα σημεία:**τα αντίστοιχα κείμενα έχουν παρόμοιο νοήμα

**Εφαρμογή σε δύο σενάρια:** Η παραπάνω λογική εφαρμόζεται και για τα δύο κείμενα που μας δίνονται στην εκφώνηση της άσκησης.

