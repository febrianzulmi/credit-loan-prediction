# %% [markdown]
# # Rakamin x ID/X Partners VIX End-to-end Solution for Credit Loan

# %% [markdown]
# ## Data Understanding + Data Preparation
#
# __Masalah__: Sebagai Data Scientist di ID/X Partners, Saya akan terlibat dalam sebuah proyek dari perusahaan pemberi pinjaman (multifinance), dimana client Saya ingin meningkatkan keakuratan dalam menilai dan mengelola risiko kredit, sehingga dapat mengoptimalkan keputusan bisnis mereka dan mengurangi potensi kerugian.
#
#
# __Metrik Bisnis__: Kerugian (Bad), margin laba bersih (Good).
#
# __Penjelasan solusi__: Kami akan mengembangkan sebuah model machine learning yang dapat mengidentifikasi potensi risiko pada pinjaman. Dengan menggunakan algoritma non-parametrik, kami akan menciptakan sebuah alat pengambilan keputusan investasi yang handal tanpa perlu terlalu banyak asumsi statistik yang rumit. Dengan model ini, kami akan dapat mengurangi investasi pada pinjaman berisiko, mengurangi kerugian, dan meningkatkan margin laba bersih kami secara signifikan
#
# __Data__ : Data pinjaman kredit antara tahun 2007 - 2014

# %%
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt  # graph
import seaborn as sns  # cool graph
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np  # linear algebra
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


filepath = 'loan_data_2007_2014.csv'

# %%
raw_data = pd.read_csv(filepath, low_memory=False)

# %%
raw_data.info()

# %% [markdown]
# Seperti yang bisa kita lihat, ada banyak data kosong dan hilang, terutama di bagian terakhir kolom. Kita harus menghapus kolom ini, dan "Unnamed: 0" yang disadap yang merupakan salinan indeks.

# %% [markdown]
# As we can see, there's many null and missing data, especially in the last bit of columns. We should drop these columns, and the bugged  which is a copy of an index.

# %%
# print the name of columns with missing values (so I can copy paste :D)
missing_values = raw_data.isnull().mean()
missing_values[missing_values == 1].index

# %%
# Obvious column to drop because it contains nothing
drop_col = ['Unnamed: 0', 'annual_inc_joint', 'dti_joint', 'verification_status_joint',
            'open_acc_6m', 'open_il_6m', 'open_il_12m', 'open_il_24m',
            'mths_since_rcnt_il', 'total_bal_il', 'il_util', 'open_rv_12m',
            'open_rv_24m', 'max_bal_bc', 'all_util', 'inq_fi', 'total_cu_tl',
            'inq_last_12m']

data = raw_data.drop(columns=drop_col, axis=1)

# %%
# loan status is our target data, so we want to check it
data.loan_status.value_counts()

# %% [markdown]
# Kita ingin memprediksi apakah suatu pinjaman berisiko atau tidak, sehingga kita perlu mengetahui akhir dari setiap pinjaman secara historis, apakah gagal bayar/ditagih, atau sudah lunas. Seperti yang bisa kita lihat, ada nilai-nilai seperti "Current", "In Grace Period" yang bersifat ambigu. Berakhirnya pinjaman tersebut dapat ditagihkan atau dilunasi seluruhnya, jadi kami tidak dapat menggunakan status tersebut. Terlambat juga agak ambigu, tapi saya pribadi tidak ingin berinvestasi pada pinjaman yang terlambat, jadi saya akan mengklasifikasikannya sebagai bad loan.
#
# Kami akan mengklasifikasikan akhir pinjaman sebagai berikut:
# - Non risky loans / good loans = ["Fully Paid", "Does not meet the credit policy. Status:Fully Paid"]
# - Risky loans / bad loans = ["Charged Off", "Late (31-120 days)", "Late (16-30 days)", "Default", "Does not meet the credit policy. Status:Charged Off"]

# %%
# define values
ambiguous = ['Current', 'In Grace Period']
good_loan = ['Fully Paid',
             'Does not meet the credit policy. Status:Fully Paid']

# drop rows that contain ambiguous ending
data = data[data.loan_status.isin(ambiguous) == False]

# create new column to classify ending
data['loan_ending'] = np.where(
    data['loan_status'].isin(good_loan), 'good', 'bad')

# %%
# check the balance
plt.title('good vs bad loans balance')
sns.barplot(x=data.loan_ending.value_counts().index,
            y=data.loan_ending.value_counts().values)

# %% [markdown]
# ## Column understanding & data leakage

# %%
data.columns

# %% [markdown]
# Berikut pemahaman saya untuk beberapa kolom:
# - 'id': The loan ID
# - 'member_id': ID peminjam disediakan. Satu anggota dapat terhubung ke lebih dari satu ID pinjaman.
# - 'loan_amnt': Jumlah pinjaman. Kolom ini sangat terkait dengan 'funded_amnt' dan 'funded_amnt_inv'
# - 'term': loan term
# - 'int_rate': interest rate
# - 'installment': Kalau di Indonesia namanya 'cicilan'. Harus terkait dengan 'loan_amnt' dan 'int_rate'. cicilan = loan_amnt * int_rate (estimasi).
# - 'grade': Nilai pinjaman
# - ... dsb
#
# >##### 1. Columns related to the loan's basic characteristic:
# 'id', 'loan_amnt', 'funded_amnt', 'funded_amnt_inv', 'term', 'int_rate', 'installment', 'grade', 'sub_grade', 'application_type'
# >
# >##### 2. Columns related to the borrower's basic identity:
# 'member_id', 'emp_title', 'emp_length', 'home_ownership', 'annual_inc', 'verification_status', 'zip_code', 'addr_state', 'dti'
# >
# >##### 3. Columns related to the borrower's personal records:
# 'delinq_2yrs', 'earliest_cr_line', 'inq_last_6mths', 'mths_since_last_delinq', 'mths_since_last_record', 'open_acc', 'pub_rec', 'revol_bal', 'revol_util', 'total_acc', 'initial_list_status', 'mths_since_last_major_derog', 'acc_now_delinq'
# >
# >##### 4. Columns related to the current status of the loan (after it is issued):
# 'issue_d', 'loan_status', 'pymnt_plan', 'out_prncp', 'out_prncp_inv', 'total_pymnt', 'total_pymnt_inv', 'total_rec_prncp',  'total_rec_int', 'total_rec_late_fee', 'recoveries', 'collection_recovery_fee', 'last_pymnt_d', 'last_pymnt_amnt', 'next_pymnt_d'

# %% [markdown]
# _Pemahaman data & kolom penting di sini_. Kami ingin memprediksi apakah suatu pinjaman berisiko atau tidak, sebelum kami berinvestasi pada pinjaman tersebut, bukan setelahnya. Soal data kita ada pada kolom terkait status pinjaman saat ini (**4**). Data kolom-kolom tersebut hanya bisa kita peroleh setelah pinjaman diberikan, dengan kata lain, setelah kita berinvestasi pada pinjaman tersebut.
#
# ~*Tidak ada gunanya mengetahui apakah suatu pinjaman berisiko atau tidak setelah kita berinvestasi di dalamnya*~
#
# Misal 'out_prncp' (pokok terhutang), bila out_prncp 0 berarti pinjaman sudah lunas, mudah diprediksi berdasarkan variabel yang satu ini saja, dan akan sangat akurat. Contoh lainnya adalah 'pemulihan', pemulihan hanya terjadi setelah peminjam tidak mampu membayar kembali pinjamannya dan lembaga pemberi pinjaman memulai proses pemulihan pinjaman. Pinjaman itu jelek dan beresiko tentunya kita tahu hanya dari info ini saja. Bagaimana variabel-variabel tersebut dapat memprediksi dengan akurat? Karena itu sudah terjadi. _Saya dapat memprediksi dengan akurasi 100% jika pertandingan sepak bola sudah berakhir._
#
# Dalam ilmu data, variabel semacam ini disebut *Kebocoran Data*. Data yang tidak akan kita dapatkan saat kita menggunakan model dalam penerapan. Kita tidak akan tahu apakah akan ada biaya pemulihan, atau apakah pokok terutang akan menjadi 0 atau tidak sebelum pinjaman selesai. Kami tidak akan mendapatkan data apa pun sebelum kami berinvestasi dalam pinjaman. Jika kita membuat model menggunakan data dengan Kebocoran Data, model kita tidak akan berguna dalam produksi.
#
# Jadi, kami akan menghilangkan kolom-kolom yang berisi Kebocoran Data dan kami hanya akan menyimpan kolom dengan data yang dapat diperoleh sebelum pinjaman diinvestasikan.

# %%
leakage_col = ['issue_d', 'loan_status', 'pymnt_plan', 'out_prncp', 'out_prncp_inv', 'total_pymnt', 'total_pymnt_inv',
               'total_rec_prncp', 'total_rec_int', 'total_rec_late_fee', 'recoveries', 'collection_recovery_fee',
               'last_pymnt_d', 'last_pymnt_amnt', 'next_pymnt_d']

data.drop(columns=leakage_col, axis=1, inplace=True)

# %%
# check duplicated data suspect based on common sense
data[['loan_amnt', 'funded_amnt', 'funded_amnt_inv',
      'grade', 'sub_grade', 'desc', 'purpose', 'title']]

# %% [markdown]
# Mari kita hapus lebih lanjut kolom-kolom yang berlebihan. Seperti yang bisa kita lihat, 'loan_amnt', 'funded_amnt', dan 'funded_amnt_inv' terlihat mirip, jadi kita bisa menghapus 2 di antaranya. 'grade' dan 'sub_grade' juga serupa, tetapi untuk memilih mana yang akan dihapus kita akan melihat nilai uniknya terlebih dahulu karena keduanya merupakan kolom kategorikal.
#
# Kolom lain yang perlu dihapus adalah 'id', 'member_id', 'url', dan 'desc'. Tiga kolom pertama masing-masing memiliki nilai unik, dan apa pun id Anda, Anda tidak akan mendapatkan perlakuan khusus apa pun, jadi itu hanya gangguan untuk model kita. Sedangkan untuk desc, cukup sulit untuk dimodifikasi, dan mengandung banyak null. Saya berasumsi bahwa kolom 'desc' adalah opsional dan banyak orang tidak mengisinya karena 'judul' dan 'tujuan' sudah cukup untuk menggambarkan pinjaman.

# %%
data[['loan_amnt', 'funded_amnt', 'funded_amnt_inv']].describe()

# %%
# based on the output, the data is so similar, and we can remove 2 of them + the other columns explained above.
drop_col = ['funded_amnt', 'funded_amnt_inv', 'id', 'member_id', 'url', 'desc']
dropped_data = data[drop_col]

data.drop(columns=drop_col, axis=1, inplace=True)

# %% [markdown]
# ## Exploring Data Analysis+ Feature Engineering

# %%
data.info()

# %% [markdown]
# __1. Personal records data__

# %% [markdown]
# Salah satu hal yang menarik berdasarkan info data di atas adalah data terkait catatan pribadi peminjam banyak sekali yang nilainya null. Khususnya, bulan-bulan sejak pelanggaran terakhir, bulan-bulan sejak catatan publik terakhir, dan bulan-bulan sejak penghinaan besar terakhir. Mari kita periksa datanya.

# %%
personal_record = ['mths_since_last_delinq',
                   'mths_since_last_record', 'mths_since_last_major_derog']

data[personal_record]

# %% [markdown]
# Seperti yang bisa kita lihat dari datanya, itu NaN atau beberapa bulan. Saya berasumsi bahwa jika mereka tidak memiliki catatan publik yang buruk, datanya tidak akan diperbarui.
#
# Agak sulit untuk mengubah data semacam ini. Nilai NaN seharusnya tidak terhingga, karena mereka adalah orang baik dan tidak pernah melakukan hal buruk. Ini juga agak berlawanan dengan intuisi, karena '0' berarti peminjam melakukan hal buruk di bulan ini (berisiko), dan semakin besar nilainya, semakin baik. Jika kita ingin mengukur seberapa besar risikonya, kita dapat menggunakan data ini sebagai penyebut suatu angka, jadi, bulan yang lebih rendah -> hasil yang lebih besar (lebih berisiko). Masalahnya, kita tidak tahu harus memilih apa sebagai pembilangnya.
#
# > Berikut ini beberapa idenya: gunakan 1 sebagai pembilang, dan (bulan+1) sebagai penyebut, hitung semua datanya, lalu ganti NaN dengan 0, sehingga nilainya akan dari 0 (tidak pernah melakukan hal buruk apa pun) menjadi 1 (baru-baru ini melakukan sesuatu yang buruk).
#
# Untungnya, 'mths_since_last_delinq' terkait dengan 'delinq_2yrs' dan 'acc_now_delinq', dan mths_since_last_record terkait dengan 'pub_rec', jadi kita cukup membuang kedua kolom tersebut. Untuk 'mths_since_last_major_derog', kami hanya akan memodifikasinya menjadi 'pernah melakukan penghinaan besar? ya (1) atau tidak (0)', untuk saat ini.

# %%
# ever did a major derogatory? 0 = nope, 1 = yes.
data['major_derogatory'] = np.where(
    data['mths_since_last_major_derog'].isna(), 0, 1)

# dropping cols
drop_col = ['mths_since_last_delinq',
            'mths_since_last_record', 'mths_since_last_major_derog']
dropped_data = pd.concat([dropped_data, data[drop_col]], axis=1)

data.drop(columns=drop_col, axis=1, inplace=True)

# %% [markdown]
# __2. Variabel yang agak kabur__

# %% [markdown]
# Kolom menarik lainnya adalah tot_coll_amt, tot_cur_bal, total_rev_hi_lim. Itu adalah tiga kolom terakhir, dan memiliki jumlah nilai bukan nol yang sama. Saya berasumsi bahwa itu adalah fitur-fitur baru antara tahun 2007 - 2014, sehingga banyak nilai yang masih nol, terutama yang lama. Deskripsi kolom-kolom ini juga tidak jelas, jadi saya agak ragu apakah datanya bocor atau tidak. Pertama-tama kita akan mencoba menjelajahi kolom-kolom ini lebih lanjut, sehingga kita dapat memutuskan apa yang harus dilakukan.

# %%
cols = ['tot_coll_amt', 'tot_cur_bal', 'total_rev_hi_lim']

# pivot table aggregated by mean
print(pd.pivot_table(data, index='loan_ending', values=cols))

# pivot table aggregated by max value
print(pd.pivot_table(data, index='loan_ending', values=cols, aggfunc=np.max))

# %%
# based on the pivot table, tot_coll_amt is kinda suspicious, let's check it
data[cols].describe()

# %%
plt.figure(figsize=(10, 6))

# I use "> 0" because 75% of the data is 0... so the plot below just use < 25% of the data
sns.kdeplot(data=data[(data['tot_coll_amt'] < 100000) & (
    data['tot_coll_amt'] > 0)], x='tot_coll_amt', hue='loan_ending')

# %%
plt.figure(figsize=(10, 6))
sns.kdeplot(data=data[data['tot_cur_bal'] < 800000],
            x='tot_cur_bal', hue='loan_ending')

# %%
plt.figure(figsize=(10, 6))
sns.kdeplot(data=data[data['total_rev_hi_lim'] < 250000],
            x='total_rev_hi_lim', hue='loan_ending')

# %% [markdown]
# Inilah yang kami temukan sejauh ini:
# - Deskripsi kolom-kolom ini agak kabur
# - 75% dari total_coll_amt adalah 0
# - Berdasarkan plot distribusi, tidak ada pemisah yang jelas antara pinjaman yang baik dan yang berisiko untuk setiap nilai kolom
# - Kita harus mengorbankan hampir ~50 ribu baris data jika ingin menggunakannya
#
# Untuk putusannya, kami memutuskan untuk tidak menggunakan kolom untuk sementara waktu.

# %%
drop_col = ['tot_coll_amt', 'tot_cur_bal', 'total_rev_hi_lim']

dropped_data = pd.concat([dropped_data, data[drop_col]], axis=1)
data.drop(drop_col, inplace=True, axis=1)

# %% [markdown]
# __3. Data dengan nilai unik kecil__

# %%
# Filtering data with less than 10 unique values
data.nunique()[data.nunique() < 10].sort_values()

# %%
# Well, we know that we should just ditch policy_code and application_type. They only have 1 value lol.
data.drop(['policy_code', 'application_type'], inplace=True, axis=1)

# %% [markdown]
# Untuk data yang nilai uniknya kecil, kita dapat mengeksplorasinya secara visual menggunakan rasio bad loan untuk setiap kategori.

# %%


def risk_pct_chart(x):
    ratio = (data.groupby(x)['loan_ending']  # group by
             .value_counts(normalize=True)  # calculate the ratio
             .mul(100)  # multiply by 100 to be percent
             .rename('risky_pct')  # rename column as percent
             .reset_index())

    sns.lineplot(data=ratio[ratio['loan_ending'] == 'bad'], x=x, y='risky_pct')
    plt.title(x)
    plt.show()


# %%
# print so I can copy paste, and remove loan_ending column :D
print(data.nunique()[data.nunique() < 10].sort_values().index)

# %%
small_unique = ['term', 'initial_list_status', 'major_derogatory',
                'verification_status', 'home_ownership', 'acc_now_delinq', 'grade',
                'collections_12_mths_ex_med']

for cols in small_unique:
    risk_pct_chart(cols)

# %% [markdown]
# What can we conclude here?
#
# __columns with significant change of good x risky ratio between values__:
# - Grade
# - Term
# - acc_now_delinq
#
# __columns with minor change of ratio between values__:
# - home ownership
# - verification status
# - major derogatory
# - initial_list_status
#
# But they're all good though, and we want to keep them, because at least the contributing something, either it's minor or major.

# %% [markdown]
# __4. Numerical vs categorical__
#
# __Cleaning__

# %%
# numerical
num_data = data.select_dtypes(exclude='object')
num_data.columns

# %%
# categorical
cat_data = data.select_dtypes(include='object')
cat_data.columns

# %% [markdown]
# Berdasarkan kolom-kolom di atas, kita dapat melihat bahwa pada kolom-kolom kategorikal, ada beberapa data yang tampaknya tidak sesuai. emp_length (panjang pekerjaan) harus berupa angka, dan 'earliest_cr_line' (batas kredit paling awal), 'last_credit_pull_d' (pemeriksaan kredit terakhir) harus berupa tanggal waktu.

# %%
cols = ['emp_length', 'earliest_cr_line', 'last_credit_pull_d']

cat_data[cols].head()

# %% [markdown]
# Asumsi saya berdasarkan data:
# - Durasi kerja yang lebih tinggi berarti pekerjaan yang lebih stabil, dan mengurangi risiko pinjaman
# - Semakin awal batas kredit berarti rekam jejak kredit yang lebih baik
# - Tanggal penarikan kredit terakhir -> pertanyaan sulit terakhir, jadi semakin lama rentang waktu antara pertanyaan (dan 'hari ini'), semakin baik. Lebih dari 2 tahun lebih baik.
#
# Kita harus memeriksanya

# %%
# Check unique value to map
data['emp_length'].unique()

# %%
emp_map = {
    '< 1 year': '0',
    '1 year': '1',
    '2 years': '2',
    '3 years': '3',
    '4 years': '4',
    '5 years': '5',
    '6 years': '6',
    '7 years': '7',
    '8 years': '8',
    '9 years': '9',
    '10+ years': '10'
}

data['emp_length'] = data['emp_length'].map(emp_map).fillna('0').astype(int)
data['emp_length'].unique()

# %%
# Pick just year from earliest credit line
data['earliest_cr_yr'] = pd.to_datetime(
    data['earliest_cr_line'], format="%b-%y").dt.year

# calculate year since last inquiry
data['yr_since_last_inq'] = 2016 - \
    pd.to_datetime(data['last_credit_pull_d'], format="%b-%y").dt.year

data[['emp_length', 'earliest_cr_yr', 'yr_since_last_inq']].describe()

# %% [markdown]
# Seperti yang bisa kita lihat, dari tahun kredit paling awal, beberapa data melebihi waktu saat ini, dengan maksimal 2068 :)
#
# Hal ini terjadi karena pd.to_datetime menggunakan 'unix' (epoch) sebagai asal atau 1970, jadi tidak ada tanggal sebelum tahun 1970, dan tanggal sebelum tahun 1970, misal. 1969, 1968, dijadikan 2068, 2067, dst.

# %%
data = data[data['earliest_cr_yr'] < 2016]
# I use 2016 as the filter because the data is from 2007 - 2014, so the latest credit line should be around 2014-2015.

# %%
to_chart = ['emp_length', 'earliest_cr_yr', 'yr_since_last_inq']

for cols in to_chart:
    risk_pct_chart(cols)

# %% [markdown]
# Kesimpulan:
# - Lama bekerja mempunyai variabilitas, namun kurang dari 1 tahun mempunyai persentase risiko paling besar.
# - Seperti yang kami duga sebelumnya, semakin awal batas kredit, semakin stabil catatan peminjam, dan kami melihat tren peningkatan risiko dalam hal ini.
# - Sejalan dengan asumsi kami, 10 - 11 tahun sejak penyelidikan terakhir memiliki akun dengan risiko paling rendah, dan 2 tahun atau kurang adalah yang tertinggi kedua. Saya tidak mengerti mengapa 9 tahun adalah yang tertinggi.

# %%
to_drop = ['earliest_cr_line', 'last_credit_pull_d']
dropped_data = pd.concat([dropped_data, data[to_drop]], axis=1)

# numerical
num_data = data.drop(to_drop, axis=1).select_dtypes(exclude='object')
print('num data: ', num_data.columns)

# categorical
cat_data = data.drop(to_drop, axis=1).select_dtypes(include='object')
print('cat data: ', cat_data.columns)

# end of numerical vs categorical cleaning :D

# %% [markdown]
# __Yang harus dilakukan__
# 1. Numerical data:
# - Histogram (distribution)
# - Correlation plot
# - Pivot
#
# 2. Categorical:
# - Balance
# - Pivot

# %%
# 1. distribution
for i in num_data.columns:
    plt.hist(num_data[i])
    plt.title(i)
    plt.show()

# %%
# 2. correlation
plt.figure(figsize=(14, 6))
sns.heatmap(data=num_data.corr(), annot=True)

# %%
# 3. pivot table
pd.pivot_table(data, index='loan_ending', values=num_data.columns)

# %% [markdown]
# Kesimpulan:
# - Hanya sejumlah kecil data numerik yang berdistribusi normal
# - Beberapa data mengandung outlier
# - Seperti yang diharapkan, jumlah angsuran & pinjaman berkorelasi hampir sempurna. Itu karena cicilan = jumlah_pinjaman * tingkat_bunga. Meskipun jumlah pinjaman dapat bervariasi, suku bunga biasanya tidak terlalu bervariasi.
#
# Berdasarkan tabel pivot, karakteristik pinjaman berisiko:
# - Berdasarkan catatan pribadi yang buruk:
#      - akun delinq lebih tinggi
#      - tunggakan lebih tinggi dalam 2 tahun terakhir
#      - permintaan yang lebih tinggi dalam 6 bulan terakhir -> permintaan yang sulit dapat mempengaruhi penilaian kredit
#      - tahun lebih rendah sejak permintaan terakhir -> lebih rendah = baru saja menerima permintaan kredit
# - Berdasarkan kesulitan pembayaran yang lebih sulit
#      - Pendapatan tahunan lebih rendah
#      - Rasio Hutang terhadap Pendapatan (DTI) lebih tinggi -> DTI = Angsuran Bulanan / Pendapatan Bulanan
#      - angsuran & jumlah pinjaman lebih tinggi
#      - tingkat bunga yang lebih tinggi (biasanya berkorelasi dengan tingkat pinjaman)
# - revol_util lebih rendah (?)
# - koleksi lebih tinggi ex med (?)

# %%
# cleaning again... I am sorry for being messy guys
cat_data.nunique()

# %% [markdown]
# Kita tidak dapat menggunakan "zip_code", "title" dan emp_title", karena mengandung terlalu banyak nilai unik sebagai variabel kategori, dan juga sulit untuk dijelajahi. Judul adalah judul pinjaman yang diberikan oleh pengguna, dan emp_title adalah judul pekerjaan mereka. "zip_code terkait dengan "addr_state", judul terkait dengan "purpose", dan emp_title terkait dengan annual_income", jadi kami dapat menghapusnya dengan aman.

# %%
to_drop = ['zip_code', 'title', 'emp_title']
# It's my habit to collect dropped data
dropped_data = pd.concat([dropped_data, data[to_drop]], axis=1)

cat_data.drop(to_drop, axis=1, inplace=True)

# %%
cat_data.columns

# %%
to_chart = ['grade', 'sub_grade', 'home_ownership',
            'verification_status', 'purpose', 'addr_state']

for cols in to_chart:
    plt.figure(figsize=(14, 4))
    risk_pct_chart(cols)

# %% [markdown]
# Kesimpulan:
# - Grade dan subgrade sesuai yang diharapkan, semakin rendah grade maka semakin berisiko pinjamannya
# - Ini agak menarik dan berlawanan dengan intuisi untuk kepemilikan rumah & status verifikasi. "None" dan "Not verified" memiliki persentase pinjaman risiko terendah.
# - Untuk purpose, 'car', 'major_purchase', dan 'wedding'  memiliki risiko terendah, dan 'small_business' memiliki risiko tertinggi
# - Menarik juga bagi negara-negara bagian untuk memiliki persentase risiko yang bervariasi.
#
# Bagaimanapun, kita akan mengubah semua data kategorikal menjadi data numerik, dan karena grade dan subgrade sama, saya akan menghapus subgrade untuk mengurangi jumlah total kolom.

# %%
dropped_data = pd.concat([dropped_data, data['sub_grade']], axis=1)
cat_data.drop('sub_grade', axis=1, inplace=True)
cat_data.nunique()

# %% [markdown]
# Inilah yang akan kami lakukan,
#
# 1. Untuk 'term' kita hanya akan menghapus "months" dan secara otomatis akan menjadi variabel numerik
# 2. Untuk 'grade' kita akan menggunakan encoder ordinal
# 3.  One hot encoding untuk:
#     - home_ownership
#     - verification status
#     - purpose
#     - addr_state
#     - initial_list_status -> tetapi hanya 1 yang cukup jadi kita akan menghilangkan 1 kolom tiruan

# %%
# 1. transforming 'term'
cat_data['term'] = cat_data['term'].str.replace(' months', '').astype(int)

# %%
cat_data['grade'].unique()

# %%
# 2. transforming grade
grade_map = {
    'A': 1,
    'B': 2,
    'C': 3,
    'D': 4,
    'E': 5,
    'F': 6,
    'G': 7,
}

cat_data['grade'] = cat_data['grade'].map(grade_map)

# %%
# 3. one hot encode?
to_dummies = ['home_ownership', 'verification_status',
              'purpose', 'addr_state', 'initial_list_status']

dummies = pd.get_dummies(cat_data[to_dummies])
dummies.drop('initial_list_status_w', axis=1, inplace=True)

# %%
dummies.head()

# %%
# dropping columns that already one hot encoded
dropped_data = pd.concat([dropped_data, cat_data[to_dummies]], axis=1)
cat_data.drop(to_dummies, axis=1, inplace=True)

# %%
# combining categorical data with one hot encoded data
cat_data_f = pd.concat([cat_data, dummies], axis=1)

# %%
# combining numerical and categorical data
final_data = pd.concat([num_data, cat_data_f], axis=1).dropna(
).reset_index().drop('index', axis=1)
final_data.head()

# %%
# separate dependant (y) and independant (X) variable
X = final_data.drop('loan_ending', axis=1)
y = final_data['loan_ending']

# %%
# splitting for training and model validation, it's important to avoid overfitting

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

# %%
train_y = np.where(train_y == 'good', 1, 0)
val_y = np.where(val_y == 'good', 1, 0)

# %% [markdown]
# # Data Modelling

# %% [markdown]
# Pokoknya saya suka datanya apa adanya, jadi kita tidak akan melakukan normalisasi/standarisasi, kita hanya akan menggunakan model ML yang tidak memerlukan distribusi/asumsi tertentu

# %% [markdown]
# __Base validation performance__

# %%
# Models

# Evaluation

# %%
# what if we accept all loan? monkey brain mode, don't need ML.
pred_y = np.where(val_y == 'good', 1, 1)
print(classification_report(val_y, pred_y))

# %% [markdown]
# Bagaimana kita membaca laporan klasifikasi ini?
#
# - _Precision_ = jika presisi > 0,5, berarti variabel prediksi yang benar lebih tinggi dari variabel prediksi yang salah. Misalnya, presisi 'good' sebesar 0,78 berarti bahwa dari semua pinjaman yang diprediksi 'good', 78% benar, dan 22% benar-benar merupakan pinjaman bad.
# - _recall_ = jika recall > 0.5, berarti berdasarkan nilai sebenarnya prediksi kita lebih dari 50% benar. Misalnya, recall 'good' sebesar 1 berarti kita memperkirakan semua pinjaman 'good' dengan benar.
#
# Untuk menyederhanakan:
# - presisi 'good' yang lebih tinggi berarti kita berinvestasi pada pinjaman yang bad lebih sedikit (investasikan pinjaman yang good > investasikan pinjaman yang bad)
# - presisi 'bad' yang lebih tinggi berarti kita lebih menghindari pinjaman bad (hindari pinjaman bad > hindari pinjaman good)
# - x% penarikan kembali 'good' berarti kami berinvestasi dalam x% pinjaman good yang tersedia (investasikan pinjaman good > hindari pinjaman good)
# - x% penarikan 'bad' berarti kita menghindari x% dari pinjaman bad yang tersedia (hindari pinjaman bad > investasikan pinjaman bad)
#
# Akurasi tidak akan memberi tahu kita apa pun. Lihat, ketika kita menggunakan otak monyet dan menerima semua pinjaman yang tersedia, keakuratannya adalah 78% haha.
#
# Evaluator mana yang harus diprioritaskan tergantung pada preferensi Anda. Dalam hal ini:
# - Presisi 'good' secara harafiah berarti keseimbangan portofolio kita, jadi saya menginginkannya setidaknya lebih tinggi dibandingkan jika kita menggunakan otak monyet (78%)
# - Saya orang yang berhati-hati, tapi, saya juga tidak mau ketinggalan banyak hal karenanya, jadi setidaknya presisi 'bad' lebih tinggi dari 50% -> untuk setiap 2 pinjaman yang saya hindari, setidaknya 1 benar-benar bad.
# - Saya tidak ingin melewatkan banyak hal, jadi saya ingin recall yang 'good' dimaksimalkan. -> berinvestasi sebanyak mungkin pada pinjaman good yang tersedia.
# - Untuk recall yang 'bad', saya tidak terlalu peduli, asalkan presisi 'bad' lebih tinggi dari 50%.

# %%
# Decision tree
dt = tree.DecisionTreeClassifier()
dt.fit(train_X, train_y)
pred_y = dt.predict(val_X)
print(classification_report(val_y, pred_y))

# %%
# KNN
knn = KNeighborsClassifier()
knn.fit(train_X, train_y)
pred_y = knn.predict(val_X)
print(classification_report(val_y, pred_y))

# %%
# Random Forest
rf = RandomForestClassifier()
rf.fit(train_X, train_y)
pred_y = rf.predict(val_X)
print(classification_report(val_y, pred_y))

# %%
# Predict probabilities
probs = rf.predict_proba(val_X)
pred_y = probs[:, 1]

# Calculate fpr, tpr, thresholds and AUC
fpr, tpr, thresholds = roc_curve(val_y, pred_y)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2,
         label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()

# %% [markdown]
# Evaluasi kurva dan nilai AUC ini untuk menilai apakah model Random Forest cenderung mengalami underfitting atau overfitting. Semakin tinggi nilai AUC, semakin baik performa model. Kurva ROC yang lebih dekat ke sudut kiri atas juga menunjukkan performa yang lebih baik.

# %%
# Logistic Regression
lr = LogisticRegression()
lr.fit(train_X, train_y)
pred_y = lr.predict(val_X)
print(classification_report(val_y, pred_y))

# %%
# Predict probabilities
probs = lr.predict_proba(val_X)
pred_y = probs[:, 1]

# Calculate fpr, tpr, thresholds and AUC
fpr, tpr, thresholds = roc_curve(val_y, pred_y)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2,
         label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()

# %% [markdown]
# Evaluasi kurva dan nilai AUC ini untuk menilai apakah model Logistic Regression cenderung mengalami underfitting atau overfitting. Semakin tinggi nilai AUC, semakin baik performa model. Kurva ROC yang lebih dekat ke sudut kiri atas juga menunjukkan performa yang lebih baik.

# %%
# ensemble soft voting classifier
voting_clf = VotingClassifier(
    estimators=[('knn', knn), ('rf', rf), ('lr', lr)], voting='soft')
voting_clf.fit(train_X, train_y)
pred_y = voting_clf.predict(val_X)
print(classification_report(val_y, pred_y))

# %% [markdown]
# Oke, jadi, kita punya 3 kandidat yang menjanjikan (berdasarkan kriteria saya sendiri) sebelum kita menyempurnakannya.
# 1. Random Forest
# 2. Logistic Regression
# 3. Soft voting classifier (_pada dasarnya, kami menggunakan setiap model dalam kasus ini_)
#
# kriteria pemfilteran saya:
# - __Harus dimiliki__: presisi 'good' > 0,78, presisi 'bad' > 0,5
# - dan untuk memilih model terakhir -> maksimalkan perolehan yang 'good'

# %% [markdown]
# ## Conclusion

# %% [markdown]
# Itu dia, model terbaik yang bisa kita miliki untuk masalah pinjaman kredit ini adalah:
# - __Random forest__: Jika Anda memilih Random forest sebagai algoritme, Dari semua pinjaman bad loan yang sebenarnya, model hanya bisa mengenali sebagian kecilnya (12%). Dari semua pinjaman yang diklasifikasikan sebagai bad loan oleh model, sebagian besar dari mereka sebenarnya adalah pinjaman baik (60%).Sebaliknya, model sangat baik dalam mengenali pinjaman baik. Hampir semua pinjaman baik yang sebenarnya dapat diidentifikasi oleh model (98%). Akurasi secara keseluruhan adalah sekitar 79%, yang artinya sebagian besar prediksi model benar.
# - __Logistic Regression__: Jika Anda memilih Logistic Regression sebagai algoritme, Dari semua pinjaman bad loan yang sebenarnya, model hanya bisa mengenali sebagian kecilnya (7%). Dari semua pinjaman yang diklasifikasikan sebagai bad loan oleh model, sebagian besar dari mereka sebenarnya adalah pinjaman baik (51%).Sebaliknya, model sangat baik dalam mengenali pinjaman baik. Hampir semua pinjaman baik yang sebenarnya dapat diidentifikasi oleh model (98%). Akurasi secara keseluruhan adalah sekitar 79%, yang artinya sebagian besar prediksi model benar.

# %%
dropped_data.columns

# %% [markdown]
# Oke, jadi, abaikan 5 kolom terakhir (satu kolom hot encoded) dari data yang dihilangkan, ada beberapa saran untuk proyek selanjutnya dengan "Credit Loan Data" ini.
# 1. __Lakukan pengambilan sampel berlebihan__. Alasannya adalah karena kita memiliki data yang cukup tidak seimbang, hanya 22% data yang merupakan pinjaman berisiko.
# 2. __Coba ubah pengelompokan status_pinjaman__. Mungkin Anda tidak keberatan dengan akhir hari 16-30, atau Anda hanya ingin menggunakan "charged off" dan "default" sebagai akhir yang buruk. Terserah Anda, dan akan menarik untuk melihat perbedaan hasilnya.
# 3. __Tambahkan kolom "berikan informasi lengkap"__. Dari kolom di atas, cukup menarik untuk melihat variabel seperti 'desc', 'zip_code', 'title', 'emp_title'. Mereka memiliki terlalu banyak nilai unik sehingga saya tidak menggunakannya, tapi, menurut saya cukup menarik untuk melihat apa pengaruh pemberian informasi tersebut terhadap risiko pinjaman. Misal: buat kolom 'provide_desc', dengan nilai 1 (pinjaman memberikan deskripsi) dan 0 (pinjaman tidak memberikan deskripsi). Kita dapat melakukannya untuk kolom berikut: 'desc', 'zip_code', 'title', 'emp_title'. Kita cukup menggunakan np.where untuk ini. Asumsi saya berdasarkan ini -> lebih lengkap informasi yang diberikan = lebih banyak niat membayar pinjaman. Khusus untuk pinjaman “small business” (memiliki pinjaman yang lebih berisiko dibandingkan kelompok lain), jika mereka memberikan gambaran, setidaknya kita tahu bahwa mereka memiliki rencana bisnis.

# %% [markdown]
# That's it! Saya merasa begitu diberkati bisa belajar begitu banyak saat melakukannya. Terima kasih kepada Partner ID/X dan Rakamin atas pencerahannya dalam dunia Machine Learning dan ilmu data serta sungguh memberikan wawasan yang berharga bagi perjalanan saya.
