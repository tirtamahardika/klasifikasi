library(caret)
library(tm)
library(SnowballC)
library(arm)
# Training data.
data <- c('Bom mapolres surabaya adalah pengalihan isu.', 'Fans Liverpool mengadakan demo di depan kantor dubes spanyol di Indonesia', 'Serbuan Tenaga Kerja China ke Indonesia yang berjumlah 10 juta', 'Gerakan Rush Money',  'Keterangan Pers soal Gaji ke-13 dan THR', 'Heboh Jackie Chan Masuk Islam.', 'Bahaya Cool Fever yang menempel di Dahi', 'Pesan Berantai Korban Tragedi Mina', 'Campuran Udang dan Vitamin C Dapat Menjadi Racun yang Berbahaya.', 'Berita Pancing Hujan Menggunakan Air Garam di Baskom',
          'Peledakan bom mapolres surabaya menelan 5 korban jiwa.', 'Serbuan Tenaga Kerja China ke Indonesia hanya ribuan', 'Rush Money merupakan kebohongan.', 'Demo di depan kantor dubes spanyol di Indonesia dibatalkan', 'Kementerian Keuangan Memastikan bahwa Keterangan Pers tersebut tidak benar.',  'Tidak ada konfirmasi resmi dari pihak Jackie Chan mengenai kepindahan dia masuk Islam.', 'dokter di Indonesia dan Malaysia menyangkal berita tersebut dan mengatakan tidak benar.', 'Korban tersebut memang korban tragedi Mina namun yang terjadi pada tahun 2004.', 'keracunan akibat mengkonsumsi udang dan vitamin C dalam waktu berdekatan tidaklah benar.', 'Air garam dalam baskom itu tidak bisa membuat uap dan menyebabkan hujan turun.')
corpus <- VCorpus(VectorSource(data))

# Create a document term matrix.
tdm <- DocumentTermMatrix(corpus, list(removePunctuation = TRUE, stopwords = TRUE, stemming = TRUE, removeNumbers = TRUE))

# Convert to a data.frame for training and assign a classification (factor) to each document.
train <- as.matrix(tdm)
train <- cbind(train, c(0, 1))

train
colnames(train)[ncol(train)] <- 'y'
train
train <- as.data.frame(train)
train$y <- as.factor(train$y)

# Train.
fit <- train(y ~ ., data = train, method = 'bayesglm')

# Check accuracy on training.
predict(fit, newdata = train)

# Test data.
data2 <- c('Jatuh korban jiwa dalam ledakan bom di mapolres surabaya.', 'Demo tersebut ditunggangi aktor politik', 'bank kehabisan dana tunai yang mengakibatkan sistem perbankan menjadi kacau.', '20 juta tenaga kerja China yang datang ke Indonesia.', 'Kemenkeu akan memberikan Gaji ke-13 dan THR',
           'Jumlah Tenaga Kerja asal China sekitar 21.000.', 'Jackie sedang dalam acara pemberian gelar Datuk kepadanya di Malaysia.', 'Demo di depan kantor dubes spanyol di Indonesia dibatalkan karena kesalahpahaman', 'Udang dan vitamin C boleh dimakan secara berdekatan.', 'Air garam dalam baskom itu tidak bisa membuat uap dan menyebabkan hujan turun.')
corpus <- VCorpus(VectorSource(data2))
tdm <- DocumentTermMatrix(corpus, control = list(dictionary = Terms(tdm), removePunctuation = TRUE, stopwords = TRUE, stemming = TRUE, removeNumbers = TRUE))
test <- as.matrix(tdm)

# Check accuracy on test.
predict(fit, newdata = test)
