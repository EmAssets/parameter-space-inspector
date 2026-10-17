"""Data for model_ladder.py: one concept (red) in 14 senses, in four languages, plus fact and context prompts.
The colour word in every sentence is marked [like this]; the marker is removed and its span recorded.
Translations are the author's (Claude's), not a native speaker's; they carry sense labels, not exact pairs."""

SENSES = {
 "colour": ["She picked a ripe [red] apple from the tree.", "The barn was painted a deep [red] last summer.", "He wore a [red] scarf against the cold.",
   "The tomatoes turned [red] on the windowsill.", "A [red] kite was stuck in the branches.", "Her new bicycle is bright [red] with white wheels.",
   "The children used a [red] crayon to draw the roof.", "The book had a faded [red] cover.", "Strawberries become [red] as they ripen.", "He mixed [red] paint with white to make pink."],
 "sky_light": ["The sky glowed [red] as the sun went down.", "A [red] moon rose during the eclipse.", "Mars appears [red] because of iron oxide dust.",
   "Betelgeuse is a [red] supergiant star.", "At dawn the clouds were streaked with [red].", "A [red] sky at night is a sailor's delight.",
   "Light at the [red] end of the spectrum has the longest wavelength.", "The volcano lit the night with a [red] glow.", "Smoke from the fires turned the sunset [red].",
   "Distant galaxies look [red] because their light is stretched."],
 "traffic": ["The driver stopped when the light turned [red].", "Never cross the road while the signal is [red].", "She ran a [red] light and got a ticket.",
   "Stop signs are [red] so drivers notice them.", "The cars waited at the [red] light for a minute.", "A flashing [red] signal means stop and then proceed with care.",
   "The pedestrian waited for the [red] man to change to green.", "Trains must halt at a [red] signal.", "He was fined for driving through a [red] light.", "When the light is [red], every lane must stop."],
 "danger": ["The alarm flashed [red] when the pressure rose.", "A [red] warning label marks the toxic chemicals.", "Her refusal to answer was a [red] flag.",
   "The engine light glowed [red] on the dashboard.", "The beach flew a [red] flag because of dangerous currents.", "Fire extinguishers are painted [red] so they are easy to find.",
   "The forecast issued a [red] alert for floods.", "The fuel gauge needle was in the [red] zone.", "The [red] lines on the report show the most urgent risks.", "The warning light turned [red] just before the crash."],
 "love": ["He gave her a dozen [red] roses on Valentine's Day.", "She sent a card covered in [red] hearts.", "The love letter was sealed with [red] wax.",
   "A [red] heart emoji ended her message.", "In poems, [red] roses stand for passionate love.", "The shop windows filled with [red] hearts every February.",
   "His gift of [red] roses told her how he felt.", "The romantic film's poster was full of [red] roses and candlelight.", "She wore [red] lipstick for their anniversary dinner.", "The couple tied a [red] ribbon on the lovers' bridge."],
 "anger": ["When he saw the dent in his car, he saw [red].", "Her face went [red] with rage.", "The insult made him see [red].", "He turned [red] with anger and slammed the door.",
   "She was [red] in the face after the argument.", "The coach went [red] and shouted at the referee.", "His ears burned [red] with fury.",
   "The unfair decision made the whole crowd see [red].", "She saw [red] when they lied to her again.", "He grew [red] and furious as the meeting dragged on."],
 "finance": ["The company has been in the [red] for three years.", "After the bad quarter, the accounts were deep in the [red].", "Her bank balance slipped into the [red].",
   "Losses were marked in [red] ink on the ledger.", "The airline's books stayed in the [red] during the pandemic.", "On the Wall Street screens, falling stocks were shown in [red].",
   "The project ran in the [red] until the second year.", "The budget ended the year in the [red] by two million dollars.", "The startup is still in the [red] but growing fast.", "The city's transit system operates in the [red]."],
 "politics_left": ["The Soviet Union flew the [red] flag with a hammer and sickle.", "Communist parties chose [red] as the colour of revolution.", "The workers sang under [red] banners on May Day.",
   "The socialist rally was a sea of [red] flags.", "Soviet troops under the [red] banner reached Berlin in 1945.", "During the Cold War, Americans feared the [red] menace.",
   "Mao's little [red] book was printed in huge numbers.", "Lenin's revolution painted the country [red].", "The party's [red] star appeared on every tank.", "He was accused of being a [red] during the McCarthy era."],
 "politics_us": ["Texas has voted [red] in presidential elections for decades.", "On the election map, Republican states are shown in [red].", "Wyoming is a reliably [red] state.",
   "Republican candidates wore [red] ties at the convention.", "The county flipped from blue to [red] in the last election.", "The [red] wave the party hoped for never came.",
   "Pollsters watched whether the state would turn [red].", "Rural districts in Ohio are mostly [red].", "Commentators split the country into [red] states and blue states.", "The governor won easily in a deeply [red] state."],
 "china_luck": ["At Chinese New Year, children receive [red] envelopes full of money.", "In China, [red] is the colour of good luck.", "Lanterns of [red] paper lined the streets for the Spring Festival.",
   "Chinese families paste [red] couplets on their doors for the new year.", "Brides in China traditionally wear [red] on their wedding day.", "Shops in Beijing hang [red] banners to bring good fortune.",
   "Firecrackers wrapped in [red] paper are lit to scare away bad luck.", "Grandmother gave him a [red] envelope for luck.", "The restaurant in Shanghai was decorated in lucky [red] and gold.",
   "Wearing [red] underwear in your zodiac year is said to bring luck in China."],
 "india_bridal": ["A married Hindu woman wears [red] sindoor in the parting of her hair.", "The Indian bride wore a [red] silk saree embroidered with gold.", "In India, [red] is the traditional colour of the bridal dress.",
   "She applied a [red] bindi before the wedding ceremony.", "The [red] bangles on her wrists showed she was newly married.", "Her [red] lehenga glittered as she walked to the mandap.",
   "At the temple, the priest marked his forehead with [red] kumkum.", "Bengali brides wear [red] and white saris on their wedding day.",
   "During the wedding, the groom applies [red] sindoor to the bride's hair.", "Wedding halls in Delhi were filled with [red] and gold decorations."],
 "death_remembrance": ["At Akan funerals in Ghana, mourners wear [red] and black cloth.", "On Remembrance Day, people wear [red] poppies for the fallen soldiers.",
   "In South Africa, [red] is sometimes worn as a colour of mourning.", "In Korea, writing a living person's name in [red] ink is considered bad luck.",
   "The mourners in Kumasi dressed in [red] for the funeral.", "The wreath of [red] poppies honoured those who died in the war.", "Bright [red] and black fabrics are traditional at Ghanaian funerals.",
   "The chief wore [red] cloth to mourn the elder.", "Wreaths of [red] poppies are laid at war memorials in November.", "Names of the dead were once written in [red] ink in Korea."],
 "sport_vehicles": ["He bought a [red] Ferrari after the deal.", "Fire engines are painted bright [red].", "The referee showed him a [red] card and sent him off.", "Liverpool fans filled the stadium in [red].",
   "The [red] sports car roared down the highway.", "Ferrari's racing cars are famous for their [red] paint.", "Manchester United play in [red] shirts.",
   "A [red] double-decker bus passed the London shop.", "The team in [red] scored in the last minute.", "The [red] racing bike won the time trial."],
 "body": ["Our [red] blood cells carry oxygen around the body.", "The skin around the cut was [red] and swollen.", "Her eyes were [red] from crying.", "A [red] rash spread across his arm.",
   "The infection made his throat [red] and sore.", "Blood looks [red] because of haemoglobin.", "His cheeks were [red] after the run.", "The burn left a [red] mark on her hand.",
   "The doctor checked the [red] patch on his knee.", "His sunburned shoulders were bright [red]."],
}

TRANS = {
 "zh": {
  "colour": ["她从树上摘了一个熟透的[红]苹果。", "去年夏天，谷仓被漆成了深[红]色。", "他围着一条[红]围巾御寒。", "窗台上的番茄变[红]了。"],
  "sky_light": ["太阳落山时，天空泛着[红]光。", "月食期间升起了一轮[红]月亮。", "火星看起来是[红]色的，因为表面有氧化铁尘埃。", "参宿四是一颗[红]超巨星。"],
  "traffic": ["信号灯变[红]时，司机停了下来。", "信号灯是[红]色时千万不要过马路。", "她闯了[红]灯，被开了罚单。", "停车标志是[红]色的，这样司机才能注意到。"],
  "danger": ["压力升高时，警报闪着[红]光。", "有毒化学品上贴着[红]色警告标签。", "她拒绝回答，这亮起了[红]灯。", "仪表盘上的发动机灯亮起了[红]色。"],
  "love": ["情人节那天他送了她一打[红]玫瑰。", "她寄了一张印满[红]心的卡片。", "那封情书用[红]色火漆封着。", "她的消息最后是一个[红]心表情。"],
  "anger": ["看到车上的凹痕，他气得满脸通[红]。", "她气得脸都[红]了。", "这句侮辱让他气得脸[红]脖子粗。", "他气得脸[红]，摔门而去。"],
  "finance": ["这家公司已经连续三年出现[赤]字。", "糟糕的季度过后，账目出现了巨额[赤]字。", "政府预算今年又是[赤]字。", "账本上的亏损用[红]墨水标出。"],
  "politics_left": ["苏联飘扬着带有锤子和镰刀的[红]旗。", "共产党选择[红]色作为革命的颜色。", "五一劳动节，工人们在[红]旗下歌唱。", "社会主义集会上是一片[红]旗的海洋。"],
  "politics_us": ["得克萨斯州几十年来一直是共和党的[红]州。", "在选举地图上，共和党州用[红]色表示。", "怀俄明州是一个稳固的[红]州。", "共和党候选人在大会上系着[红]领带。"],
  "china_luck": ["春节时，孩子们会收到装满钱的[红]包。", "在中国，[红]色是好运的颜色。", "春节期间，街道两旁挂满了[红]灯笼。", "中国家庭过年时在门上贴[红]对联。"],
  "india_bridal": ["已婚的印度教女性会在发缝里点上[红]色的朱砂。", "这位印度新娘穿着绣金的[红]色丝绸纱丽。", "在印度，[红]色是新娘礼服的传统颜色。", "婚礼前，她在额头点了一颗[红]色吉祥痣。"],
  "death_remembrance": ["在加纳阿坎人的葬礼上，吊唁者穿着[红]黑相间的布。", "在阵亡将士纪念日，人们佩戴[红]色罂粟花悼念阵亡士兵。", "在南非，[红]色有时被当作哀悼的颜色。", "在韩国，用[红]墨水写活人的名字被认为不吉利。"],
  "sport_vehicles": ["交易完成后，他买了一辆[红]色法拉利。", "消防车被漆成鲜[红]色。", "裁判向他出示了[红]牌，把他罚下场。", "利物浦球迷身穿[红]衣挤满了球场。"],
  "body": ["[红]细胞把氧气输送到全身。", "伤口周围的皮肤又[红]又肿。", "她的眼睛哭[红]了。", "他的胳膊上起了一片[红]疹。"]},
 "hi": {
  "colour": ["उसने पेड़ से एक पका हुआ [लाल] सेब तोड़ा।", "पिछली गर्मियों में खलिहान को गहरे [लाल] रंग से रंगा गया।", "ठंड से बचने के लिए उसने [लाल] मफलर पहना।", "खिड़की पर रखे टमाटर [लाल] हो गए।"],
  "sky_light": ["सूरज ढलते समय आसमान [लाल] चमक उठा।", "ग्रहण के दौरान [लाल] चाँद निकला।", "आयरन ऑक्साइड की धूल के कारण मंगल [लाल] दिखाई देता है।", "बीटलजूस एक [लाल] महादानव तारा है।"],
  "traffic": ["बत्ती [लाल] होते ही ड्राइवर रुक गया।", "सिग्नल [लाल] हो तो सड़क कभी पार न करें।", "उसने [लाल] बत्ती पार की और चालान कट गया।", "स्टॉप साइन [लाल] होते हैं ताकि ड्राइवर उन्हें देख सकें।"],
  "danger": ["दबाव बढ़ने पर अलार्म [लाल] चमकने लगा।", "ज़हरीले रसायनों पर [लाल] चेतावनी लेबल लगा है।", "उसका जवाब देने से इनकार करना एक [लाल] झंडी थी।", "डैशबोर्ड पर इंजन की बत्ती [लाल] जल उठी।"],
  "love": ["वैलेंटाइन डे पर उसने उसे एक दर्जन [लाल] गुलाब दिए।", "उसने [लाल] दिलों से भरा कार्ड भेजा।", "प्रेम पत्र को [लाल] मोम से सील किया गया था।", "उसके संदेश के अंत में [लाल] दिल वाला इमोजी था।"],
  "anger": ["अपनी गाड़ी में डेंट देखकर वह गुस्से से [लाल] हो गया।", "गुस्से से उसका चेहरा [लाल] हो गया।", "उस अपमान से उसकी आँखें गुस्से से [लाल] हो गईं।", "वह गुस्से से [लाल] होकर दरवाज़ा पटककर चला गया।"],
  "finance": ["शेयर बाज़ार आज [लाल] निशान में बंद हुआ।", "खराब तिमाही के बाद कंपनी के खाते [लाल] निशान में चले गए।", "निवेशकों के पोर्टफोलियो [लाल] निशान में हैं।", "बही-खाते में घाटे को [लाल] स्याही से लिखा गया।"],
  "politics_left": ["सोवियत संघ हँसिया-हथौड़े वाला [लाल] झंडा फहराता था।", "कम्युनिस्ट पार्टियों ने क्रांति के रंग के रूप में [लाल] को चुना।", "मई दिवस पर मज़दूरों ने [लाल] झंडों के नीचे गीत गाए।", "रैली में कॉमरेडों ने [लाल] सलाम किया।"],
  "politics_us": ["टेक्सास दशकों से राष्ट्रपति चुनावों में [लाल] राज्य रहा है।", "चुनावी नक्शे पर रिपब्लिकन राज्य [लाल] रंग में दिखाए जाते हैं।", "वायोमिंग एक पक्का [लाल] राज्य है।", "रिपब्लिकन उम्मीदवारों ने सम्मेलन में [लाल] टाई पहनी।"],
  "china_luck": ["चीनी नव वर्ष पर बच्चों को पैसों से भरे [लाल] लिफ़ाफ़े मिलते हैं।", "चीन में [लाल] रंग सौभाग्य का रंग है।", "वसंत उत्सव के लिए सड़कों पर [लाल] कागज़ की लालटेनें लगी थीं।", "चीनी परिवार नए साल पर अपने दरवाज़ों पर [लाल] पट्टियाँ चिपकाते हैं।"],
  "india_bridal": ["विवाहित हिंदू महिला अपनी माँग में [लाल] सिंदूर लगाती है।", "भारतीय दुल्हन ने सोने की कढ़ाई वाली [लाल] रेशमी साड़ी पहनी।", "भारत में दुल्हन का पारंपरिक जोड़ा [लाल] रंग का होता है।", "शादी की रस्म से पहले उसने [लाल] बिंदी लगाई।"],
  "death_remembrance": ["घाना में अकान लोगों के अंतिम संस्कार में शोक मनाने वाले [लाल] और काले कपड़े पहनते हैं।", "स्मृति दिवस पर लोग शहीद सैनिकों की याद में [लाल] पॉपी लगाते हैं।", "दक्षिण अफ़्रीका में [लाल] रंग कभी-कभी शोक के रंग के रूप में पहना जाता है।", "कोरिया में किसी जीवित व्यक्ति का नाम [लाल] स्याही से लिखना अशुभ माना जाता है।"],
  "sport_vehicles": ["सौदे के बाद उसने एक [लाल] फ़रारी खरीदी।", "दमकल गाड़ियाँ चटक [लाल] रंग की होती हैं।", "रेफ़री ने उसे [लाल] कार्ड दिखाकर मैदान से बाहर भेज दिया।", "लिवरपूल के प्रशंसक [लाल] कपड़ों में स्टेडियम में भरे थे।"],
  "body": ["हमारी [लाल] रक्त कोशिकाएँ पूरे शरीर में ऑक्सीजन पहुँचाती हैं।", "कटे के आसपास की त्वचा [लाल] और सूजी हुई थी।", "रोने से उसकी आँखें [लाल] हो गई थीं।", "उसकी बाँह पर [लाल] चकत्ते फैल गए।"]},
 "es": {
  "colour": ["Ella cogió una manzana [roja] y madura del árbol.", "El granero fue pintado de [rojo] oscuro el verano pasado.", "Llevaba una bufanda [roja] contra el frío.", "Los tomates se pusieron [rojos] en la ventana."],
  "sky_light": ["El cielo se volvió [rojo] cuando se puso el sol.", "Una luna [roja] salió durante el eclipse.", "Marte se ve [rojo] por el polvo de óxido de hierro.", "Betelgeuse es una supergigante [roja]."],
  "traffic": ["El conductor se detuvo cuando el semáforo se puso en [rojo].", "Nunca cruces la calle con el semáforo en [rojo].", "Se saltó un semáforo en [rojo] y le pusieron una multa.", "Las señales de stop son [rojas] para que los conductores las vean."],
  "danger": ["La alarma parpadeó en [rojo] cuando subió la presión.", "Una etiqueta de advertencia [roja] marca los productos tóxicos.", "Su negativa a responder fue una bandera [roja].", "La luz del motor se encendió en [rojo] en el salpicadero."],
  "love": ["Le regaló una docena de rosas [rojas] el día de San Valentín.", "Ella envió una tarjeta llena de corazones [rojos].", "La carta de amor estaba sellada con lacre [rojo].", "Un emoji de corazón [rojo] cerraba su mensaje."],
  "anger": ["Cuando vio la abolladura en su coche, se puso [rojo] de ira.", "Su cara se puso [roja] de rabia.", "El insulto lo puso [rojo] de furia.", "Se puso [rojo] de ira y dio un portazo."],
  "finance": ["La empresa lleva tres años en números [rojos].", "Tras el mal trimestre, las cuentas quedaron en números [rojos].", "Su cuenta bancaria entró en números [rojos].", "Las pérdidas se anotaron con tinta [roja] en el libro de cuentas."],
  "politics_left": ["La Unión Soviética ondeaba la bandera [roja] con la hoz y el martillo.", "Los partidos comunistas eligieron el [rojo] como color de la revolución.", "Los obreros cantaron bajo banderas [rojas] el Primero de Mayo.", "El mitin socialista era un mar de banderas [rojas]."],
  "politics_us": ["Texas ha votado [rojo] en las elecciones presidenciales durante décadas.", "En el mapa electoral, los estados republicanos aparecen en [rojo].", "Wyoming es un estado [rojo] seguro.", "Los candidatos republicanos llevaban corbatas [rojas] en la convención."],
  "china_luck": ["En el Año Nuevo chino, los niños reciben sobres [rojos] llenos de dinero.", "En China, el [rojo] es el color de la buena suerte.", "Faroles de papel [rojo] adornaban las calles en la Fiesta de la Primavera.", "Las familias chinas pegan pareados [rojos] en sus puertas por Año Nuevo."],
  "india_bridal": ["Una mujer hindú casada lleva sindoor [rojo] en la raya del pelo.", "La novia india llevaba un sari de seda [roja] bordado en oro.", "En la India, el [rojo] es el color tradicional del vestido de novia.", "Se puso un bindi [rojo] antes de la ceremonia de boda."],
  "death_remembrance": ["En los funerales akan de Ghana, los dolientes visten telas [rojas] y negras.", "En el Día del Recuerdo, la gente lleva amapolas [rojas] por los soldados caídos.", "En Sudáfrica, el [rojo] se lleva a veces como color de luto.", "En Corea, escribir el nombre de una persona viva con tinta [roja] trae mala suerte."],
  "sport_vehicles": ["Se compró un Ferrari [rojo] después del trato.", "Los camiones de bomberos están pintados de [rojo] brillante.", "El árbitro le mostró la tarjeta [roja] y lo expulsó.", "Los aficionados del Liverpool llenaron el estadio de [rojo]."],
  "body": ["Nuestros glóbulos [rojos] llevan oxígeno por todo el cuerpo.", "La piel alrededor del corte estaba [roja] e hinchada.", "Tenía los ojos [rojos] de tanto llorar.", "Un sarpullido [rojo] se extendió por su brazo."]},
}

COLOURS8 = ["red", "blue", "green", "yellow", "black", "white", "orange", "purple"]
FACTS_RED = ["The colour of fresh blood is", "Ripe strawberries are", "A stop sign is painted", "The surface of Mars looks",
  "At a traffic light, the colour that means stop is", "Fire engines are usually painted", "Ferrari sports cars are famous for being",
  "In China, the envelopes of money given at New Year are", "Communist flags are usually", "On US election maps, Republican states are coloured",
  "Ripe tomatoes are", "A ruby is a gemstone that is", "When you are embarrassed, your cheeks turn", "Valentine's Day roses are traditionally",
  "Hot chili peppers are often", "The colour of a male cardinal bird is", "The Coca-Cola logo is white on", "The maple leaf on the Canadian flag is",
  "In accounting, losses are written in", "A referee sends a player off by showing a card that is", "Indian brides traditionally wear the colour",
  "The Swiss flag has a white cross on a background of", "The circle on the flag of Japan is", "Cherries are", "When lobsters are cooked, they turn",
  "The longest wavelength of visible light looks", "A warning light that means danger is usually", "Santa Claus wears a suit that is", "The Soviet flag was", "Ladybugs are usually"]

FACTS_OTHER = [("The clear daytime sky is", "blue"), ("Fresh grass is", "green"), ("Ripe bananas are", "yellow"), ("Fresh snow is", "white"), ("Coal is", "black"),
  ("The deep ocean looks", "blue"), ("Leaves in summer are", "green"), ("Lemons are", "yellow"), ("Milk is", "white"), ("Crows are", "black"),
  ("Ripe oranges are", "orange"), ("Violets are often", "purple"), ("Blueberries are", "blue"), ("Limes are", "green"), ("School buses in America are", "yellow")]

# composition (M5): the same context with two colours must give two different meanings. {c} = the colour word.
# Neither the context alone nor the colour alone decides; only the combination does.
CROSSED = {
 "traffic": (["At a traffic light, {c} means", "When the signal turns {c}, drivers must", "A {c} traffic light tells drivers to"], {"red": " stop", "green": " go"}),
 "us_election": (["On a US election map, {c} states vote", "In American politics, {c} is the colour of the", "A {c} state in the US usually votes"], {"red": " Republican", "blue": " Democratic"}),
 "china_dress": (["In China, the colour {c} is traditionally worn at", "In Chinese tradition, people wear {c} clothes at", "In China, {c} decorations are used for"], {"red": " weddings", "white": " funerals"}),
 "china_envelope": (["In China, {c} envelopes of money are given at", "A Chinese family hands out {c} envelopes at", "In Chinese custom, money in {c} envelopes is given at"], {"red": " weddings", "white": " funerals"}),
 "india_dress": (["In India, a woman dressed in {c} is usually", "In traditional Hindu custom, a woman wearing {c} is", "A Hindu woman in a {c} sari is traditionally"], {"red": " married", "white": " widowed"}),
 "accounts": (["In accounting, numbers in {c} show a", "When a company's figures are in {c}, it is making a", "On a balance sheet, {c} ink marks a"], {"red": " loss", "black": " profit"}),
 "stocks_china": (["On Chinese stock market screens, {c} numbers mean prices are", "In China's stock market, a {c} number shows the price is", "On a stock app in Shanghai, {c} means the share price is"], {"red": " rising", "green": " falling"}),
 "stocks_us": (["On American stock market screens, {c} numbers mean prices are", "On Wall Street, a {c} number shows the price is", "On a stock app in New York, {c} means the share price is"], {"red": " falling", "green": " rising"}),
 "football": (["In football, a {c} card means the player is", "The referee showed a {c} card, so the player was", "A {c} card in soccer means the player has been"], {"red": " sent", "yellow": " warned"}),
 "wine": (["A glass of {c} wine is usually served with", "With {c} wine, a chef will usually serve", "The best food to pair with {c} wine is"], {"red": " meat", "white": " fish"}),
 "ribbon": (["A {c} ribbon is worn to raise awareness of", "People wear a {c} ribbon to support awareness of", "The {c} ribbon is the symbol of the fight against"], {"red": " AIDS", "pink": " breast"}),
 "beach": (["On a beach, a {c} flag means the water is", "When lifeguards fly a {c} flag, the sea is", "A {c} beach flag tells swimmers the water is"], {"red": " dangerous", "green": " safe"}),
 "ship": (["On a ship, the {c} light marks the", "At night, a boat shows a {c} light on its", "Sailors know that the {c} navigation light is on the"], {"red": " port", "green": " starboard"}),
}

# M0 / M8 word lists
RELATED = ["crimson", "scarlet", "blood", "rose", "fire", "ruby", "cherry", "maroon", "stop", "danger", "pink", "orange", "apple", "tomato", "flame",
           "anger", "love", "communist", "Republican", "warning", "wine", "heart", "strawberry", "lipstick", "Ferrari"]
FOREIGN = ["红", "赤", "लाल", "rojo", "rouge", "rot", "красный", "أحمر", "rosso", "vermelho", "kırmızı", "merah"]
ASSOC = ["blood", "fire", "love", "danger", "stop", "apple", "rose", "anger", "communism", "luck", "China", "wine", "heart", "cherry", "strawberry",
         "Ferrari", "Mars", "tomato", "ruby", "lipstick", "flag", "alarm", "sunset", "embarrassment", "Republican", "wedding", "passion", "warning",
         "brick", "lobster", "ocean", "grass", "snow", "sky", "banana", "ice", "cloud", "cucumber", "piano", "mathematics"]
COMMON = """time year people way day man thing woman life child world school state family student group country problem hand part place case week company
system program question work government number night point home water room mother area money story fact month lot right study book eye job word business
issue side kind head house service friend father power hour game line end member law car city community name president team minute idea kid body
information back parent face others level office door health person art war history party result change morning reason research girl guy moment air
teacher force education foot boy age policy process music market sense nation plan college interest death experience effect use class control care field
development role effort rate drug show leader light voice wife police mind price report decision son view relationship town road arm difference
value building action model season society tax director position player record paper space ground form event official matter center couple site project
activity star table need court oil situation cost industry figure street image phone data picture practice piece land product doctor wall patient worker
news test movie north support technology step baby computer type attention film tree source organization hair window evidence population camera
chair glass river garden bread stone engine bottle pencil ticket island summer winter animal horse bird fish salt sugar metal plastic""".split()

COLOR_FRAMES = ["The wall was painted {w}.", "She wore a {w} dress to the party.", "The sky turned {w} at dusk.", "He bought a {w} car last week.", "The flower was a bright shade of {w}."]
COLORS = {"red": "#ff0000", "orange": "#ffa500", "yellow": "#ffff00", "green": "#008000", "blue": "#0000ff", "purple": "#800080",
          "pink": "#ffc0cb", "brown": "#a52a2a", "black": "#000000", "white": "#ffffff", "grey": "#808080", "cyan": "#00ffff",
          "magenta": "#ff00ff", "violet": "#ee82ee", "crimson": "#dc143c", "scarlet": "#ff2400", "teal": "#008080", "navy": "#000080",
          "gold": "#ffd700", "maroon": "#800000", "lime": "#00ff00", "turquoise": "#40e0d0", "beige": "#f5f5dc", "olive": "#808000"}

# ── methods v2 (model_ladder.py --methods 2). Nothing above changes; v1 runs never read these. ──

# M5 crossed design: mirrored context pairs. Each pair uses the same two colours and the same two meanings, with the
# mapping swapped between its contexts, so a colour -> meaning association that ignores the context scores the same
# DiD in both halves and an interaction of 0. stocks_china / stocks_us (above) is the first pair.
CROSSED_MIRRORED = {
 "lean_us": (["On a US election map, a {c} state leans to the", "In American politics, the {c} side is on the political", "A {c} district in the US leans to the"], {"red": " right", "blue": " left"}),
 "lean_uk": (["In British politics, the {c} party leans to the", "In the UK, the {c} side is on the political", "A {c} seat in Britain leans to the"], {"red": " left", "blue": " right"}),
 "candle_china": (["On a Chinese trading chart, a {c} candle means the price went", "In Shanghai, a {c} bar on a price chart means the stock went", "On a Chinese chart, a {c} candlestick shows the price moved"], {"red": " up", "green": " down"}),
 "candle_us": (["On an American trading chart, a {c} candle means the price went", "In New York, a {c} bar on a price chart means the stock went", "On an American chart, a {c} candlestick shows the price moved"], {"red": " down", "green": " up"}),
 "temp_tap": (["The tap marked {c} gives water that is", "In a bathroom, the {c} tap runs water that is", "A {c} dot on a tap means the water is"], {"red": " hot", "blue": " cool"}),
 "temp_star": (["In astronomy, a {c} star has a surface that is relatively", "Compared with the Sun, a {c} star is", "A {c} star's surface is relatively"], {"red": " cool", "blue": " hot"}),
 "buoy_europe": (["Entering a harbour in Europe, a sailor keeps {c} buoys on the", "In European waters, a {c} channel buoy is passed on the", "Sailing into a European port, {c} markers stay on the"], {"red": " left", "green": " right"}),
 "buoy_us": (["Entering a harbour in America, a sailor keeps {c} buoys on the", "In American waters, a {c} channel buoy is passed on the", "Sailing into an American port, {c} markers stay on the"], {"red": " right", "green": " left"}),
}
# (context, mirrored context): each pair is one cluster for M5's sign-flip null
MIRROR_PAIRS = [("stocks_china", "stocks_us"), ("lean_us", "lean_uk"), ("candle_china", "candle_us"), ("temp_tap", "temp_star"), ("buoy_europe", "buoy_us")]

# English sense sentences whose sense cue comes only AFTER the marked word ("A [red] kite was stuck..."): a causal
# model's state at the marked token cannot have seen the cue. (sense, index into SENSES[sense]); the author's reading.
CUE_AFTER = [("colour", 4), ("colour", 6), ("sky_light", 1), ("sky_light", 5), ("traffic", 2), ("traffic", 5),
             ("danger", 1), ("danger", 2), ("danger", 4), ("danger", 6), ("danger", 8),
             ("love", 0), ("love", 1), ("love", 3), ("love", 4), ("love", 5), ("love", 6), ("love", 8), ("love", 9),
             ("anger", 3), ("anger", 4), ("anger", 5), ("anger", 9), ("politics_left", 9), ("politics_us", 5),
             ("china_luck", 2), ("china_luck", 7), ("china_luck", 9), ("india_bridal", 3), ("india_bridal", 4), ("india_bridal", 5),
             ("death_remembrance", 2), ("death_remembrance", 6), ("death_remembrance", 7),
             ("sport_vehicles", 0), ("sport_vehicles", 4), ("sport_vehicles", 7), ("sport_vehicles", 9), ("body", 0), ("body", 3)]

# M8 crossed design: the same 40 words (ASSOC) rated for red and for two other colours, each with its own examples.
M8_HEADS = {
 "red": "How strongly is each word associated with the colour red? 0 = not at all, 9 = extremely.\nred - cardinal: 8\nred - pillow: 2\nred - crimson: 9\nred - spoon: 1\n",
 "blue": "How strongly is each word associated with the colour blue? 0 = not at all, 9 = extremely.\nblue - sapphire: 9\nblue - pillow: 2\nblue - navy: 8\nblue - spoon: 1\n",
 "green": "How strongly is each word associated with the colour green? 0 = not at all, 9 = extremely.\ngreen - emerald: 9\ngreen - pillow: 2\ngreen - moss: 8\ngreen - spoon: 1\n",
}

# names defined for --methods 2 only: v1 code that scans this module's texts (model_ladder.control_pair,
# model_intervene.neutral_sentences) leaves them out so v1 runs keep exactly the words they had
V2_ONLY = ("CROSSED_MIRRORED", "MIRROR_PAIRS", "CUE_AFTER", "M8_HEADS", "V2_ONLY")
