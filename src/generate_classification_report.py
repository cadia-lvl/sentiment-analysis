


import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from pprint import pprint
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class SentimentDataset(Dataset):
    # Constructor Function 
    def __init__(self, reviews, targets, tokenizer, max_len):
        self.reviews = reviews
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    # Length magic method
    def __len__(self):
        return len(self.reviews)
    
    # get item magic method
    def __getitem__(self, item):
        review = str(self.reviews[item])
        target = self.targets[item]
        
        # Encoded format to be returned 
        encoding = self.tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        return {
            'review_text': review,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(target, dtype=torch.long)
        }
    
class SentimentData(Dataset):
    def __init__(self, reviews, sentiments, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.text = reviews
        self.targets = sentiments
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split())
        
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]


        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }



class RoBERTaClassificationReport:
    def __init__(self, model, tokenizer, review, sentiment, device):
        self.model = model
        self.tokenizer = tokenizer
        self.test_data = self.create_data_loader(review, sentiment, tokenizer)
        self.device = device

    def create_data_loader(self, review, sentiment, tokenizer, max_len=512, batch_size=8) -> DataLoader:
        ds = SentimentDataset(
            reviews=review.to_numpy(),
            targets=sentiment.to_numpy(),
            tokenizer=tokenizer,
            max_len=max_len
        )
    
        return DataLoader(
            ds,
            batch_size=batch_size,
            num_workers=0
        )

    def generate_report(self) -> (str | dict):
        self.model.eval()
        y_true = []
        y_pred = []
        with torch.no_grad():
            for batch in self.test_data:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['targets'].to(self.device)
                outputs = self.model(input_ids, attention_mask=attention_mask, return_dict=True)
                prediction = torch.max(outputs.logits, dim=1)
                y_true.extend(labels.tolist())
                y_pred.extend(prediction.indices.tolist())
        report = classification_report(y_true, y_pred, output_dict=True)
        return report


class DataFrameLoader():
    def __init__(self, pdf_src, sample_size=None, random_state=42):
        
        self.df = pd.read_csv(pdf_src)
        if "Unnamed: 0" in self.df.columns:
            self.df.drop(['Unnamed: 0'], axis=1, inplace=True)
        self.df['sentiment'] = self.df.sentiment.apply(lambda sentiment: 1 if sentiment == 'positive' else 0)
        
        if sample_size is not None:
            self.df = self.df.sample(n=sample_size, random_state=random_state)
        
        # 70% train, 15% validation, 15% test
        X_train, X_temp, y_train, y_temp = train_test_split(self.df["review"], self.df["sentiment"], test_size=0.3, random_state=random_state)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=random_state)
        
        self.X_all = self.df.review
        self.y_all = self.df.sentiment

        self.X_train = X_train
        self.y_train = y_train

        self.X_val = X_val
        self.y_val = y_val

        self.X_test = X_test
        self.y_test = y_test
def eval_files():
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    device = "cuda"
    data = [
        # {
        #     'folder': './roberta-batch8-unprocessed_model/',
        #     'filename': '../IMDB-Dataset.csv'
        # },
        # {
        #     'folder': './icebert-google-batch8-remove-noise-model/',
        #     'filename': '../IMDB-Dataset-GoogleTranslate.csv'
        # },
        # {
        #     'folder': './IceBERT-mideind-batch8-remove-noise-model/',
        #     'filename': '../IMDB-Dataset-MideindTranslate.csv'
        # },
        {
            'folder': './icebert-google-batch8-remove-noise-model/',
            'filename': '../Hannes-Movie-Reviews.csv'
        },
        {
            'folder': './IceBERT-mideind-batch8-remove-noise-model/',
            'filename': '../Hannes-Movie-Reviews.csv'
        }
    ]
    for d in data:
        folder = d['folder']
        filename = d['filename']
        print("Loading model from folder {} using file {}".format(folder, filename))
        dfl = DataFrameLoader(filename)
        model = AutoModelForSequenceClassification.from_pretrained(folder)
        model.to(device)
        tokenizer = AutoTokenizer.from_pretrained(folder)
        report = RoBERTaClassificationReport(model, tokenizer, dfl.X_all, dfl.y_all, device)
        pprint(report.generate_report())
        print("*"*50)

if __name__ == '__main__':
    pass


t = """Í pistli dagsins mun ég fjalla um tvær afar ólíkar myndir sem sýndar hafa verið í bíóhúsum Reykjavíkurborgar undanfarið, annars vegar finnsku hrollvekjuna Hatching eða Klakið eftir Hönnu Bergholm og bandaríska blaðamennskudramað She Said, sem hin þýska Maria Schrader leikstýrði.
Ég var ótrúlega þakklát fyrir að hafa villst inn á það sem gæti hafa verið ein af síðustu sýningunum á Klakinu í Bíó Paradís í síðustu viku, mynd sem var frumsýnd á Sundance-hátíðinni í byrjun árs en ég hafði lítið sem ekkert heyrt talað um. Myndin fangaði hug minn undir eins. Í upphafsatriðinu sjáum við móður sem heldur úti einhvers konar lífstílsbloggi eða vídjódagbók sem gefur fylgjendum hennar innsýn í líf hinnar fullkomnu fjölskyldu; hún tekur myndband af dóttur sinni æfa sig fyrir fimleikakeppni í óaðfinnanlegri stofu heimilisins, áður en faðir og yngri bróðir bætast í hópinn og brosa sínu blíðasta fyrir myndavélina. Um leið og hún hefur slökkt á upptökunni flýgur svartur fugl inn í stofuna, sem svo óheppilega vill til að er full af brothættum munum, og kemur sögusviðinu í uppnám. Fjölskyldan leggst öll á eitt í því að ná þessari skaðræðisskepnu, sem tekst að brjóta að því er virðist skuggalega marga vasa áður en dóttirin fangar hana loksins í handklæði. Andrúmsloftið, sem hefur orðið æ óþægilegra eftir því sem fuglinn flögraði lengur um, verður bókstaflega óhugnanlegt þegar móðir hennar, brosmild og fögur, tekur fuglinn af henni og snýr hann úr hálslið á augabragði.
Þetta upphaf setur tóninn fyrir myndina, sem er uppfull af sálfræðilegum óhugnaði sem við upplifum í gegnum aðalpersónu myndarinnar, hina 12 ára gömlu Tinju, sem Siiri Solalinna túlkar svo til óaðfinnanlega í sínu fyrsta hlutverki í kvikmynd. Að kvöldi þessa sama dags heyrir hún fuglsgarg fyrir utan gluggann og þegar hún fer út að athuga málið finnur hún egg sem hún tekur með sér inn og hlúir að. Mér dettur ekki í hug að spilla myndinni fyrir ykkur með því að lýsa nákvæmlega því sem klekst að lokum út úr þessu eggi, þið verðið eiginlega að sjá það sjálf. Það má þó segja að afkvæmið verði að hálfgerðri framlengingu á Tinju, sem gerir myndinni kleift að fjalla um sálarlíf stúlkunar með næmni sem sjaldgæft er að maður rekist á í hrollvekju. Í samhengi má reyndar minnast á að finnska orðið yfir það að klekja vísar einnig til hugarangurs, þess að hafa áhyggjur, þannig að Tinja klekur ekki bara út eggi í þessari ágætu mynd.
Klakið minnti mig reyndar svolítið á aðra, mun hefðbundnari hryllingsmynd frá 1976, Carrie eftir Brian de Palma. Þótt ég haldi mikið upp á Carrie, þá hefur það fyrir löngu orðið að klisju að gera kvenlíkamann óhugnanlegan í hrollvekjum, eins og þau ykkar vita sem hafa séð eina vinsælustu hryllingsmynd haustsins, Barbarian eftir Zach Cregger, á streymisveitu Disney. Klakið fellur ekki í þessa gildru. Mig grunar að þar sé leikstjóranum Hönnu Bergholm að þakka, sem hefur lýst því í viðtölum hversu þreytt hún sé á skorti á áhugverðum kvenpersónum í kvikmyndum. Óhugnaðurinn í Klakinu býr í óheilbrigðum samböndum og einmanaleika, sem er einhverra hluta vegna mjög óvenjulegt viðfangsefni innan hryllingsmyndagreinarinnar, þrátt fyrir það að líklegra sé að áhorfendur tengi við það frekar ýmislegt annað sem þeirri ágætu grein er hugleikið.
Á þessum femínísku nótum vil ég beina sjónum að mynd Mariu Schrader, She Said, þar sem vinkonurnar Carey Mulligan og Zoe Kazan fara með hlutverk samstarfs- og vinkvennana Jodi Kantor og Megan Twohey. She Said byggir á samnefndri bók eftir Kantor og Twohey frá 2019, en báðar eru þær blaðakonur hjá The New York Times. Í bókinni rekja þær aðdragandann að því að greinar þeirra um kynferðisafbrot kvikmyndaframleiðandans Harveys Weinstein birtust í blaðinu í októberbyrjun 2017. Þær Kantor og Twohey voru fyrstar til að fjalla um málið á ítarlegan hátt, með frásögnum kvenna sem komu fram undir nafni, þótt Ronan Farrow hafi fylgt þeim fast á eftir með umfjöllun í New Yorker, en hlustendur muna eflaust flestir eftir þeirri gífurlegu samfélagsumræðu sem fylgdi í kjölfarið og áhrifum hennar á vöxt og útbreiðslu #metoo-hreyfingarinnar víðsvegar um heim.
She Said er að mestu mjög hefðbundin blaðamennskumynd, sem nýtir sér hefðbundin stef slíkra kvikmynda. Við fylgjumst með metnaðarfullum blaðamönnum sem vinna hörðum höndum að því að afhjúpa sannleikann, taka símtöl hvenær sem er sólarhringsins, elta viðmælendur uppi víðsvegar um heiminn og taka stöðuna á fundum með ritstjórum blaðsins með reglubundnum hætti. Twohey og Kantor upplifa mótlæti í formi hins valdamikla og árásargjarna Weinsteins, sem reynist ítrekað hafa gert trúnaðarsamninga við fórnarlömb sín, en mikil hluti af þeirri mikilvægu vinnu sem liggur greinum þeirra að baki fólst í því að sannfæra konurnar sem hann braut á um það að rjúfa þögnina um ofbeldið. Það kemur því kannski ekki á óvart að myndir eins og All the President’s Men (1976) og Spotlight (2015) hafi töluvert borið á góma í samhengi við She Said, sem fjalla um metnaðarfulla rannsóknarblaðamennsku andspænis þöggunartilburðum bandarískra yfirvalda annars vegar og kaþólsku kirkjunnar hins vegar, og margir hafa velt vöngum yfir því hvort mynd Mariu Schrader eigi í vændum viðlíka velgengni á Óskarsverðlaununum og þessir forverar hennar frá 1976 og 2015.
Það sem greinir She Said frá öðrum þekktum blaðamannamyndum er ef til vill helst það að hún fjallar um konur; blaðakonurnar tvær og konurnar sem treystu þeim fyrir sögum sínum. Weinstein sjálfur er svo til alveg fjarverandi í myndinni, við sjáum baksvip hans þegar hann mætir á fund í höfuðstöðvar New York Times ásamt lögfræðingum sínum, en annars heyrum við einungis rödd hans í síma. Myndin dvelur líka meira við áhrif rannsóknarinnar á einkalíf Kantor og Twohey en hefð er fyrir í sambærilegum verkum, en þær eiga báðar maka og ung börn sem þeim tekst misvel að búa til tíma fyrir í öllum hasarnum. Myndin gefur þannig raunsærri mynd af lífi rannsóknarblaðamanns, sem getur ekki alltaf gefið sig rannsókninni á vald með alveg sama hætti og oft sést í slíkum myndum. She Said hefur reyndar óvenju mikinn raunsæisblæ, sem birtist til dæmis líka þeirri ákvörðun að taka upp í raunverulegum höfuðstöðvum The New York Times á Manhattan, sem taka sig eimkar vel út á hvíta tjaldinu.
Þótt helstu kostir She Said liggi að hluta í raunsæislegri nálgun á viðfangsefnið—því það er óneitanlega hressandi sjá söguhetjur sem eru að breyta heiminum á sama tíma og þær þurfa að takast á við hversdagslegri hluti eins og fæðingarþunglyndi og annað álag sem fylgir barneignum—þá er má rekja galla myndarinnar, í mínum huga, til þessarar sömu tilhneigingar. Því myndin er, eins og mörg önnur sambærileg verk, rétt rúmlega tveggja tíma löng. Viðfangsefnið er þó þess eðlis að við erum vön því að ákveðnum stílbrögðum sé beitt til að halda áhorfendum við efnið og byggja upp spennu; aðferðum sem kalla má „kvikmyndalegar“ og stangast gjarnan á við raunsæislega framsetningu. Og ég verð að viðurkenna það, að þótt ég hafi verið ánægð með myndina yfir það heila og ég mæli með henni, þá saknaði ég þess sem greinir veruleikann frá Hollywoodmyndum, eins og til dæmis tónlist sem hraðar á hjartslættinum að öðru hverju, dramatískari samtölum en gengur og gerist í hversdagslífinu og myndfléttuatriðum sem gefa tilfinningu fyrir því að fólk sé að vinna vinnuna sína hraðar og með meiri tilþrifum en ég hef nokkurn tímann getað gert sjálf."""
device = "cuda"
folder = './IceBERT-mideind-batch8-remove-noise-model/'
model = AutoModelForSequenceClassification.from_pretrained(folder)
model.to(device)
tokenizer = AutoTokenizer.from_pretrained(folder)
data={
    "sentiment": [0],
    "review": [t]
}
df = pd.DataFrame(data)

s = SentimentDataset([t], [0], tokenizer, 512)

model.eval()

batch = s[0]

with torch.no_grad():
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['targets'].to(device)

    print(labels)

    #outputs = model(input_ids, attention_mask=attention_mask, return_dict=True)
    #print(outputs)
    #prediction = torch.max(outputs.logits, dim=1)

#print(p)