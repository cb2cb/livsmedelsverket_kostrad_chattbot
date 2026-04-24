import os
import time
import pickle

import streamlit as st
from google import genai
from dotenv import load_dotenv
from pypdf import PdfReader
from google.genai import types
import numpy as np

# Ladda miljövariabler från .env-fil
load_dotenv()

# Språkväljare – högst upp så titeln och underrubriken samt disclaimern får rätt språk
if "språk" not in st.session_state:
    st.session_state.språk = "Svenska"

språk = st.selectbox("Välj språk / Choose language", [
    "Svenska", "English", "العربية", "Suomi", "Español", "Soomaali",
    "فارسی", "دری", "ܐܪܡܝܐ", "Polski", "Bosanski", "Hrvatski",
    "Srpski", "Shqip", "Dansk", "Deutsch", "Français", "हिन्दी",
    "Italiano", "Kurdî", "Nederlands", "Norsk", "Português", "Română",
    "Русский", "Tigrinya", "Türkçe", "Українська", "中文", "日本語", "한국어",
])
st.session_state.språk = språk

# ---------------------------------------------------------------
# Ladda in data (körs bara en gång tack vare @st.cache_resource)
# ---------------------------------------------------------------
class EmbeddingsSvar:
    def __init__(self, embeddings):
        self.embeddings = embeddings

@st.cache_resource
def ladda_data():
    client = genai.Client(api_key=os.getenv("GENAI_API_KEY"), vertexai=False)

    filer = [
        "bra-mat-for-spadbarn-under-ett-ar-livsmedelsverket.pdf",
        "spaedbarn_kostrad.pdf",
        "1177_barnmat.pdf",
        "bra-mat-nar-du-ar-gravid-lattlast.pdf",
        "gravida_mat.pdf",
        "gravida_naring.pdf",
        "gravida_matmangd.pdf",
        "gravida_vego.pdf",
        "barn_vegetarisk.pdf",
        "barn_vegansk.pdf",
        "barn_1_2_ar.pdf"
    ]

    text = ""
    for fil in filer:
        reader = PdfReader(fil)
        for page in reader.pages:
            text += page.extract_text()

    #Styckebaserad chunking delar på styckesgränser istället för teckenantal 
    #som jag hade i början. Detta gör att texten i varje chunk hänger bättre ihop 
    #och minskar risken för att viktiga meningar delas upp. 
    #Hade problem med att vissa chunkar inte innehöll tillräckligt med information för att ge bra svar, 
    #nu blir det mer sammanhängande text i varje chunk, vilket resulterar i bättre svar. """
    def skapa_chunks(text, max_storlek=2000, overlap=200):
        stycken = [s.strip() for s in text.split("\n\n") if s.strip()]
        chunks = []
        nuvarande_chunk = ""
        for stycke in stycken:
            if len(nuvarande_chunk) + len(stycke) < max_storlek:
                nuvarande_chunk += stycke + "\n\n"
            else:
                if nuvarande_chunk:
                    chunks.append(nuvarande_chunk.strip())
                nuvarande_chunk = nuvarande_chunk[-overlap:] + stycke + "\n\n"
        if nuvarande_chunk:
            chunks.append(nuvarande_chunk.strip())
        return chunks

    chunks = skapa_chunks(text, max_storlek=2000, overlap=200)

    if os.path.exists("embeddings.pkl"):
        with open("embeddings.pkl", "rb") as f:
            embeddings = pickle.load(f)
    else:
        alle_embeddings = []
        batch_storlek = 50
        for i in range(0, len(chunks), batch_storlek):
            batch = chunks[i:i + batch_storlek]
            svar = client.models.embed_content(
                model="models/gemini-embedding-2-preview",
                contents=batch,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
            )
            alle_embeddings.extend(svar.embeddings)
            time.sleep(15)

        embeddings = EmbeddingsSvar(alle_embeddings)
        with open("embeddings.pkl", "wb") as f:
            pickle.dump(embeddings, f)

    return client, chunks, embeddings

client, chunks, embeddings = ladda_data()

#För att undvika att skicka hundratals API-anrop varje gång appen startas sparas 
#embeddings i en lokal fil kallad embeddings.pkl med hjälp av Python-biblioteket pickle. 
#När appen startas kontrolleras det om filen finns. 
#Om den finns laddas embeddings direkt från disk, annars skapas de och sparas för framtida bruk. 
#API-anropen skickas i batchar om 50 chunks med en paus på 15 sekunder mellan varje batch, 
#för att inte överstiga Gemini API:ets hastighetsbegränsning."""

# ---------------------------------------------------------------
# Hjälpfunktioner
# ---------------------------------------------------------------
# Nedan kommer cosine similarity och semantic search som används för att hitta de mest 
# relevanta textstyckena (chunkarna). 

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def semantic_search(query, chunks, embeddings, top_k=5):
    query_response = client.models.embed_content(
        model="models/gemini-embedding-2-preview",
        contents=query,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
    )
    query_embedding = query_response.embeddings[0].values

    similarity_scores = []
    for i, chunk_embedding in enumerate(embeddings.embeddings):
        score = cosine_similarity(query_embedding, chunk_embedding.values)
        similarity_scores.append((i, score))

    similarity_scores.sort(key=lambda x: x[1], reverse=True)
    top_indices = [index for index, _ in similarity_scores[:top_k]]
    return [chunks[index] for index in top_indices]

def ask_chatbot(question, history, språk):
    relevant_chunks = semantic_search(question, chunks, embeddings, top_k=3)
    context = "\n".join(relevant_chunks)

# Nedan kommer prompten som skickas till Gemini. 
# Den innehåller instruktioner, funktion för att minnas historiken för konversationen, 
# och för att ta fram relevant information från PDF-filerna.

    historik_text = ""
    for msg in history[:-1]:
        roll = "Användare" if msg["role"] == "user" else "Assistent"
        historik_text += f"{roll}: {msg['content']}\n"

    prompt = f"""
    Du är en hjälpsam assistent för gravida, närstående till gravida, föräldrar med spädbarn eller småbarn.
    Använd endast information från PDF-filerna för att svara på frågan och ingen annan information.
    Om svaret inte finns i PDF-filerna, säg att du inte vet.
    Formulera dig enkelt och dela upp svaret i fina stycken - citera aldrig direkt från texten.
    Återge aldrig exakta meningar eller stycken från källmaterialet. Omformulera alltid informationen helt i egna ord.
    Ge alltid ett tydligt och enhetligt svar. Börja med en direkt rekommendation och förklara sedan varför.
    Om du får en fråga om barn utan specifik ålder, fråga efter specifik ålder innan du svarar.
    Undvik att säga något som motsäger dig själv i samma svar.
    Kom ihåg information från tidigare i konversationen, till exempel barnets ålder eller tidigare frågor och svar.
    Användaren har valt språk: {språk}. Svara alltid på detta språk oavsett vilket språk frågan är skriven på.
    Om användaren skriver något kort som "ok", "tack" eller liknande, svara kort och vänligt och påminn dem gärna om att de kan ställa fler frågor.

    Tidigare konversation:
    {historik_text}

    Information från PDF:
    {context}

    Fråga: {question}
    """

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"Ett fel uppstod vid anrop till Gemini: {e}"

# ---------------------------------------------------------------
# Titlar, underrubriker, placeholders och disclaimers på olika språk
# ---------------------------------------------------------------
titlar = {
    "Svenska": "Kostråd för gravida, spädbarn och småbarn (0 - 2 år)",
    "English": "Dietary advice for pregnant women, infants and toddlers (0 - 2 years)",
    "العربية": "نصائح غذائية للحوامل والرضع والأطفال الصغار (0 - 2 سنة)",
    "Suomi": "Ravitsemusneuvoja raskaana oleville, vauvoille ja taaperoille (0 - 2 vuotta)",
    "Español": "Consejos nutricionales para embarazadas, bebés y niños pequeños (0 - 2 años)",
    "Soomaali": "Talooyinka nafaqada ee haweenka uurka leh, ilmaha iyo carruurta (0 - 2 sano)",
    "فارسی": "توصیه‌های تغذیه‌ای برای زنان باردار، نوزادان و کودکان (0 - 2 سال)",
    "دری": "توصیه‌های غذایی برای زنان حامله، نوزادان و اطفال (0 - 2 سال)",
    "ܐܪܡܝܐ": "ܡܠܟܘܬܐ ܕܙܘܢܐ ܠܢܫ̈ܐ ܒܛܢ̈ܬܐ، ܝܠܘܕ̈ܐ ܘܛܠܝ̈ܐ (0 - 2 ܫܢ̈ܐ)",
    "Polski": "Porady żywieniowe dla kobiet w ciąży, niemowląt i małych dzieci (0 - 2 lata)",
    "Bosanski": "Prehrambeni savjeti za trudnice, dojenčad i malu djecu (0 - 2 godine)",
    "Hrvatski": "Prehrambeni savjeti za trudnice, dojenčad i malu djecu (0 - 2 godine)",
    "Srpski": "Prehrambeni saveti za trudnice, odojčad i malu decu (0 - 2 godine)",
    "Shqip": "Këshilla ushqimore për gratë shtatzëna, foshnjat dhe fëmijët e vegjël (0 - 2 vjet)",
    "Dansk": "Kostråd til gravide, spædbørn og småbørn (0 - 2 år)",
    "Deutsch": "Ernährungsratschläge für Schwangere, Säuglinge und Kleinkinder (0 - 2 Jahre)",
    "Français": "Conseils nutritionnels pour les femmes enceintes, les nourrissons et les jeunes enfants (0 - 2 ans)",
    "हिन्दी": "गर्भवती महिलाओं, शिशुओं और छोटे बच्चों के लिए आहार संबंधी सलाह (0 - 2 वर्ष)",
    "Italiano": "Consigli nutrizionali per donne in gravidanza, neonati e bambini piccoli (0 - 2 anni)",
    "Kurdî": "Şîretên xwarinê ji bo jinên ducanî, pitikan û zarokên piçûk (0 - 2 sal)",
    "Nederlands": "Voedingsadvies voor zwangere vrouwen, baby's en peuters (0 - 2 jaar)",
    "Norsk": "Kostråd for gravide, spedbarn og småbarn (0 - 2 år)",
    "Português": "Conselhos nutricionais para gestantes, bebês e crianças pequenas (0 - 2 anos)",
    "Română": "Sfaturi nutriționale pentru femei gravide, sugari și copii mici (0 - 2 ani)",
    "Русский": "Советы по питанию для беременных женщин, младенцев и маленьких детей (0 - 2 года)",
    "Tigrinya": "ምኽሪ ምግቢ ንጥንስቲ ደቀንስትዮ፣ ህጻናትን ንኣሽቱ ቆልዑን (0 - 2 ዓመት)",
    "Türkçe": "Hamile kadınlar, bebekler ve küçük çocuklar için beslenme tavsiyeleri (0 - 2 yaş)",
    "Українська": "Поради щодо харчування для вагітних жінок, немовлят і маленьких дітей (0 - 2 роки)",
    "中文": "孕妇、婴儿和幼儿的饮食建议（0 - 2岁）",
    "日本語": "妊婦、乳児、幼児のための栄養アドバイス（0〜2歳）",
    "한국어": "임산부, 영아 및 유아를 위한 영양 조언 (0 - 2세)",
}

underrubriker = {
    "Svenska": "Ställ frågor om mat och näring för gravida, spädbarn och småbarn. Svaren baseras på råd från Livsmedelsverket och 1177.",
    "English": "Ask questions about nutrition for pregnant women, infants and toddlers. Answers are based on guidelines from the Swedish Food Agency and 1177.",
    "العربية": "اطرح أسئلة حول التغذية للحوامل والرضع والأطفال. الإجابات مبنية على إرشادات وكالة الغذاء السويدية و1177.",
    "Suomi": "Kysy ravitsemuksesta raskaana oleville, vauvoille ja taaperoille. Vastaukset perustuvat Ruotsin elintarvikeviraston ja 1177:n ohjeisiin.",
    "Español": "Haz preguntas sobre nutrición para embarazadas, bebés y niños pequeños. Las respuestas se basan en las pautas de la Agencia Sueca de Alimentos y 1177.",
    "Soomaali": "Su'aalo wax ka weydii nafaqada haweenka uurka leh, ilmaha iyo carruurta. Jawaabaha waxay ku salaysan yihiin tilmaamaha Hay'adda Cuntada Iswidishka iyo 1177.",
    "فارسی": "سوالاتی درباره تغذیه برای زنان باردار، نوزادان و کودکان بپرسید. پاسخ‌ها بر اساس رهنمودهای آژانس غذای سوئد و 1177 است.",
    "دری": "سوالاتی درباره تغذیه برای زنان حامله، نوزادان و اطفال بپرسید. جواب‌ها بر اساس رهنمودهای اداره غذای سویدن و 1177 است.",
    "ܐܪܡܝܐ": "ܫܐܠ ܫܘ̈ܐܠܐ ܥܠ ܙܘܢܐ ܠܢܫ̈ܐ ܒܛܢ̈ܬܐ، ܝܠܘܕ̈ܐ ܘܛܠܝ̈ܐ. ܦܘܢܝ̈ܐ ܡܣܬܡܟܝܢ ܥܠ ܬܚܘܡ̈ܐ ܕܣܘܟܠܐ ܕܙܘܢܐ ܕܣܘܝܕ ܘ1177.",
    "Polski": "Zadaj pytania dotyczące żywienia kobiet w ciąży, niemowląt i małych dzieci. Odpowiedzi opierają się na wytycznych Szwedzkiej Agencji Żywności i 1177.",
    "Bosanski": "Postavite pitanja o ishrani trudnica, dojenčadi i male djece. Odgovori se temelje na smjernicama Švedske agencije za hranu i 1177.",
    "Hrvatski": "Postavite pitanja o prehrani trudnica, dojenčadi i male djece. Odgovori se temelje na smjernicama Švedske agencije za hranu i 1177.",
    "Srpski": "Postavite pitanja o ishrani trudnica, odojčadi i male dece. Odgovori se zasnivaju na smernicama Švedske agencije za hranu i 1177.",
    "Shqip": "Bëni pyetje rreth ushqyerjes për gratë shtatzëna, foshnjat dhe fëmijët e vegjël. Përgjigjet bazohen në udhëzimet e Agjencisë Suedeze të Ushqimit dhe 1177.",
    "Dansk": "Stil spørgsmål om ernæring for gravide, spædbørn og småbørn. Svarene er baseret på retningslinjer fra det svenske fødevareagentur og 1177.",
    "Deutsch": "Stellen Sie Fragen zur Ernährung für Schwangere, Säuglinge und Kleinkinder. Die Antworten basieren auf den Richtlinien der Schwedischen Lebensmittelbehörde und 1177.",
    "Français": "Posez des questions sur la nutrition pour les femmes enceintes, les nourrissons et les jeunes enfants. Les réponses sont basées sur les directives de l'Agence suédoise de l'alimentation et 1177.",
    "हिन्दी": "गर्भवती महिलाओं, शिशुओं और छोटे बच्चों के पोषण के बारे में प्रश्न पूछें। उत्तर स्वीडिश खाद्य एजेंसी और 1177 के दिशानिर्देशों पर आधारित हैं।",
    "Italiano": "Fai domande sulla nutrizione per donne in gravidanza, neonati e bambini piccoli. Le risposte si basano sulle linee guida dell'Agenzia alimentare svedese e 1177.",
    "Kurdî": "Pirsên di derheqê xwarinê de ji bo jinên ducanî, pitikan û zarokên piçûk bipirsin. Bersiv li ser rêbernameyên Ajansa Xwarinê ya Swêdê û 1177 in.",
    "Nederlands": "Stel vragen over voeding voor zwangere vrouwen, baby's en peuters. Antwoorden zijn gebaseerd op richtlijnen van het Zweedse Voedselbureau en 1177.",
    "Norsk": "Still spørsmål om ernæring for gravide, spedbarn og småbarn. Svarene er basert på retningslinjer fra det svenske Mattilsynet og 1177.",
    "Português": "Faça perguntas sobre nutrição para gestantes, bebês e crianças pequenas. As respostas são baseadas nas diretrizes da Agência Sueca de Alimentos e 1177.",
    "Română": "Puneți întrebări despre nutriția pentru femei gravide, sugari și copii mici. Răspunsurile se bazează pe ghidurile Agenției Suedeze pentru Alimente și 1177.",
    "Русский": "Задавайте вопросы о питании беременных женщин, младенцев и маленьких детей. Ответы основаны на рекомендациях Шведского агентства по продовольствию и 1177.",
    "Tigrinya": "ሕቶታት ብዛዕባ ምግቢ ንጥንስቲ ደቀንስትዮ፣ ህጻናትን ንኣሽቱ ቆልዑን ሕተቱ። መልስታት ኣብ መምርሒታት ናይ ሽወደን ናይ ምግቢ ኤጀንሲን 1177ን ዝተመርኮሱ እዮም።",
    "Türkçe": "Hamile kadınlar, bebekler ve küçük çocuklar için beslenme hakkında sorular sorun. Yanıtlar, İsveç Gıda Ajansı ve 1177'nin yönergelerine dayanmaktadır.",
    "Українська": "Ставте запитання про харчування вагітних жінок, немовлят і маленьких дітей. Відповіді ґрунтуються на рекомендаціях Шведського агентства з харчування та 1177.",
    "中文": "就孕妇、婴儿和幼儿的营养提问。答案基于瑞典食品局和1177的指导方针。",
    "日本語": "妊婦、乳児、幼児の栄養について質問してください。回答はスウェーデン食品庁と1177のガイドラインに基づいています。",
    "한국어": "임산부, 영아 및 유아의 영양에 관한 질문을 하세요. 답변은 스웨덴 식품청과 1177의 지침을 기반으로 합니다.",
}

placeholder_texter = {
    "Svenska": "Ställ din fråga här...",
    "English": "Ask your question here...",
    "العربية": "اكتب سؤالك هنا...",
    "Suomi": "Kirjoita kysymyksesi tähän...",
    "Español": "Escribe tu pregunta aquí...",
    "Soomaali": "Halkan ku qor su'aaladaada...",
    "فارسی": "سوال خود را اینجا بنویسید...",
    "دری": "سوال خود را اینجا بنویسید...",
    "ܐܪܡܝܐ": "ܫܐܠܬܟ ܟܬܘܒ ܗܪܟܐ...",
    "Polski": "Wpisz swoje pytanie tutaj...",
    "Bosanski": "Upišite svoje pitanje ovdje...",
    "Hrvatski": "Upišite svoje pitanje ovdje...",
    "Srpski": "Unesite svoje pitanje ovde...",
    "Shqip": "Shkruaj pyetjen tënde këtu...",
    "Dansk": "Stil dit spørgsmål her...",
    "Deutsch": "Stellen Sie Ihre Frage hier...",
    "Français": "Posez votre question ici...",
    "हिन्दी": "अपना प्रश्न यहाँ लिखें...",
    "Italiano": "Scrivi la tua domanda qui...",
    "Kurdî": "Pirsê xwe li vir binivîse...",
    "Nederlands": "Stel uw vraag hier...",
    "Norsk": "Still spørsmålet ditt her...",
    "Português": "Escreva sua pergunta aqui...",
    "Română": "Scrieți întrebarea dvs. aici...",
    "Русский": "Задайте свой вопрос здесь...",
    "Tigrinya": "ሕቶኻ ኣብዚ ጸሓፍ...",
    "Türkçe": "Sorunuzu buraya yazın...",
    "Українська": "Введіть своє запитання тут...",
    "中文": "在此输入您的问题...",
    "日本語": "ここに質問を入力してください...",
    "한국어": "여기에 질문을 입력하세요...",
}

disclaimers = {
    "Svenska": "⚠️ Denna app ersätter inte medicinsk rådgivning. Kontakta alltid BVC eller vården vid tveksamheter.",
    "English": "⚠️ This app does not replace medical advice. Always contact your healthcare provider if in doubt.",
    "العربية": "⚠️ هذا التطبيق لا يحل محل المشورة الطبية. تواصل دائماً مع مقدم الرعاية الصحية عند الشك.",
    "Suomi": "⚠️ Tämä sovellus ei korvaa lääketieteellistä neuvontaa. Ota aina yhteyttä terveydenhuoltoon epäselvissä tilanteissa.",
    "Español": "⚠️ Esta aplicación no reemplaza el consejo médico. Siempre consulte a su proveedor de atención médica si tiene dudas.",
    "Soomaali": "⚠️ App-kani kuma beddeli karo talooyinka caafimaadka. Had iyo jeer la xiriir bixiyaha daryeelka caafimaadka haddaad shaki qabto.",
    "فارسی": "⚠️ این برنامه جایگزین مشاوره پزشکی نمی‌شود. در صورت تردید همیشه با ارائه‌دهنده مراقبت‌های بهداشتی خود تماس بگیرید.",
    "دری": "⚠️ این برنامه جای مشوره طبی را نمی‌گیرد. در صورت شک همیشه با خدمات صحی تماس بگیرید.",
    "ܐܪܡܝܐ": "⚠️ ܗܢܐ ܐܦܠܝܩܝܫܢ ܠܐ ܡܚܠܦ ܠ ܡܠܟܘܬܐ ܕܐܣܝܘܬܐ. ܒܟܠ ܙܒܢ ܩܪܘ ܠ ܡܛܦܠܢܐ ܕܚܘܠܡܢܐ ܐܢ ܐܝܬ ܠܟܘܢ ܦܫ.",
    "Polski": "⚠️ Ta aplikacja nie zastępuje porady medycznej. W razie wątpliwości zawsze skontaktuj się z lekarzem.",
    "Bosanski": "⚠️ Ova aplikacija ne zamjenjuje medicinski savjet. Uvijek se obratite zdravstvenom radniku u slučaju sumnje.",
    "Hrvatski": "⚠️ Ova aplikacija ne zamjenjuje medicinski savjet. Uvijek se obratite zdravstvenom radniku u slučaju nedoumice.",
    "Srpski": "⚠️ Ova aplikacija ne zamenjuje medicinski savet. Uvek se obratite zdravstvenom radniku u slučaju nedoumice.",
    "Shqip": "⚠️ Ky aplikacion nuk zëvendëson këshillën mjekësore. Gjithmonë kontaktoni ofruesin tuaj të kujdesit shëndetësor nëse keni dyshime.",
    "Dansk": "⚠️ Denne app erstatter ikke medicinsk rådgivning. Kontakt altid sundhedspersonale ved tvivl.",
    "Deutsch": "⚠️ Diese App ersetzt keine medizinische Beratung. Wenden Sie sich bei Unsicherheiten immer an Ihren Arzt.",
    "Français": "⚠️ Cette application ne remplace pas les conseils médicaux. Consultez toujours votre professionnel de santé en cas de doute.",
    "हिन्दी": "⚠️ यह ऐप चिकित्सा सलाह का विकल्प नहीं है। संदेह होने पर हमेशा अपने स्वास्थ्य सेवा प्रदाता से संपर्क करें।",
    "Italiano": "⚠️ Questa app non sostituisce il consiglio medico. Consulta sempre il tuo medico in caso di dubbi.",
    "Kurdî": "⚠️ Ev app şûna şîreta bijîşkî nagire. Her gav bi peydakarê lênêrîna tenduristiyê ve têkilî daynin heke gumanên we hebin.",
    "Nederlands": "⚠️ Deze app vervangt geen medisch advies. Neem altijd contact op met uw zorgverlener bij twijfel.",
    "Norsk": "⚠️ Denne appen erstatter ikke medisinsk rådgivning. Kontakt alltid helsepersonell ved tvil.",
    "Português": "⚠️ Este aplicativo não substitui o conselho médico. Sempre consulte seu profissional de saúde em caso de dúvida.",
    "Română": "⚠️ Această aplicație nu înlocuiește sfatul medical. Contactați întotdeauna furnizorul de servicii medicale în caz de îndoială.",
    "Русский": "⚠️ Это приложение не заменяет медицинскую консультацию. При сомнениях всегда обращайтесь к врачу.",
    "Tigrinya": "⚠️ እዚ ኣፕ ናይ ሕክምና ምኽሪ ኣይትካእን። ኣብ ምጥርጣር ምስ ናይ ጥዕና ኣቕራቢ ርኸቡ።",
    "Türkçe": "⚠️ Bu uygulama tıbbi tavsiyenin yerini tutmaz. Şüphe durumunda her zaman sağlık uzmanınıza başvurun.",
    "Українська": "⚠️ Цей додаток не замінює медичної консультації. При сумнівах завжди звертайтеся до лікаря.",
    "中文": "⚠️ 本应用程序不能替代医疗建议。如有疑问，请务必咨询您的医疗保健提供者。",
    "日本語": "⚠️ このアプリは医療アドバイスの代わりにはなりません。疑問がある場合は、必ず医療提供者に連絡してください。",
    "한국어": "⚠️ 이 앱은 의료 조언을 대체하지 않습니다. 의심스러운 경우 항상 의료 제공자에게 문의하세요.",
}

# ---------------------------------------------------------------
# Chattgränssnitt
# ---------------------------------------------------------------

# Sidhuvud – anpassat efter valt språk
st.title(titlar.get(språk, titlar["Svenska"]))
st.write(underrubriker.get(språk, underrubriker["Svenska"]))

st.markdown("""
    <style>
    .stApp {
        background-color: #f8f9fa;
    }
    h1 {
        color: #2c7a5c;
        font-size: 1.6rem;
    }
    .stMarkdown p {
        color: #555555;
    }
    [data-testid="stChatMessageContent"] {
        border-radius: 12px;
        padding: 4px 8px;
    }
    .stChatInput input {
        font-size: 16px;
        border-radius: 20px;
    }
    @media (min-width: 768px) {
        .main .block-container {
            padding: 2rem 3rem;
            max-width: 800px;
        }
    }
    @media (max-width: 767px) {
        .main .block-container {
            padding: 1rem;
        }
        h1 {
            font-size: 1.2rem;
        }
    }
    </style>
""", unsafe_allow_html=True)

# Diskret disclaimer
st.caption(disclaimers.get(språk, disclaimers["Svenska"]))

st.divider()

# Spara chatthistorik mellan meddelanden
if "messages" not in st.session_state:
    st.session_state.messages = []

# Visa tidigare meddelanden
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Ta emot ny fråga från användaren
if fråga := st.chat_input(placeholder_texter.get(språk, placeholder_texter["Svenska"])):
    st.session_state.messages.append({"role": "user", "content": fråga})
    with st.chat_message("user"):
        st.write(fråga)

    with st.chat_message("assistant"):
        with st.spinner("Hämtar svar..."):
            svar = ask_chatbot(fråga, st.session_state.messages, språk)
        st.write(svar)

    st.session_state.messages.append({"role": "assistant", "content": svar})

# ---------------------------------------------------------------
# REFLEKTION: Verklig användning, möjligheter och utmaningar
# 
#
# VERKLIG ANVÄNDNING:
# Denna chattbot skulle kunna användas av exempelvis BVC (barnavårdscentraler),
# barnsjukhus eller appar riktade till nyblivna föräldrar. Istället för att
# söka igenom långa dokument kan föräldrar snabbt få svar på specifika frågor
# om barnets mat och näring – dygnet runt och utan att behöva vänta på personal.
#
# MÖJLIGHETER:
# - Tillgänglighet: Föräldrar kan få svar när som helst, även mitt i natten.
# - Språkbarriärer: Appen stöder 30+ språk vilket gör den tillgänglig för
#   nyanlända föräldrar som ännu inte behärskar svenska.
# - Skalbarhet: En bot kan hantera flertalet frågor utan extra kostnad.
# - Affärsmässigt: Kan integreras i befintliga vårdappar eller kanske säljas som tjänst
#   till regioner och kommuner.
#
# UTMANINGAR:
# - Medicinsk säkerhet: Feltolkad eller föråldrad information kan skada barn.
#   Systemet måste tydligt uppmana användare att kontakta vården vid tveksamheter.
# - Uppdatering av data: Kostråd förändras. Dokumenten måste hållas aktuella,
#   annars kan botten ge felaktig information.
# - Etik och ansvar: Vem bär ansvaret om botten ger fel råd? Tydliga
#   ansvarsfriskrivningar och mänsklig granskning är nödvändigt.
# - GDPR: Även om appen inte sparar personuppgifter skickas frågor till Googles
#   servrar.