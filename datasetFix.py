import unidecode

f = open("training_data.csv",encoding = "ISO-8859-1")
w = open("training_fix.csv",'w')

for i in f.readlines():
    w.write(unidecode.unidecode(i))
