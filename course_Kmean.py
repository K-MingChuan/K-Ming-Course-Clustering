import re
import numpy as np
import json
import jieba
import jieba.analyse
from collections import defaultdict
from sklearn.preprocessing import scale
from sklearn.cluster import KMeans
from sklearn import metrics

jieba.set_dictionary('dict.txt.big')
#jieba.analyse.set_stop_words('stopwords-zh.txt')

#sss = '[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+'
#string = re.sub(sss.decode("utf8"), "".decode("utf8"),"")
regexString = r'[^\u4e00-\u9fa5]'

jsonFileName = 'courses_106_2' #courses_106_2

def loadCourseInformationFromFile():
    fileName = 'course_UTF-8/{}.json'.format(jsonFileName)
    with open(fileName, 'r', encoding='utf-8') as f:
        courseINF = json.load(f)
    return courseINF

def getCourseInformation(courseINF):
    course = [re.sub('[0-9]', '', w)
                  for attr in courseINF
                      for w in attr.values()]
    return course

def decodeWords(course):
    words = [w for w in course]
    return words

def getJiebaWords(words):
    jiebaWords = [jieba.analyse.extract_tags(w)
                      for w in words]
    return jiebaWords

def getCourseKeyWords(wordsSet):
    wordsVector = [word for word in wordsSet]
    return wordsVector

def saveCourseWords(course_Words):
    fileName = '{}_words_vector.txt'.format(jsonFileName)
    with open(fileName, 'w', encoding='utf-8') as f:
        for word in course_Words:
            f.write(word + '\n')

def getCourseTraining_Words(courseINF):
    word = []
    for course in courseINF:
        tmp = []
        for key in course:
            if key == 'name':
                for name in [course[key]]*4: #control the influences of course
                    tmp += jieba.analyse.extract_tags(name)
            else:
                tmp += jieba.analyse.extract_tags(re.sub(r'[0-9]', '', course[key]))
        word.append(tmp)
    return word

def createCourseVector(all_courseWords):
    fileName = 'Word_Vector/{}_words_vector.txt'.format(jsonFileName)
    with open(fileName, 'r', encoding='utf-8') as f:
        wordList = [word.rstrip('\n') for word in f.readlines()]
        
        courses_Vector = []
        for words in all_courseWords:
            vector = defaultdict(float)
            for word in wordList:
                vector[word] += words.count(word)
            courses_Vector.append([vec for vec in vector.values()])
    return courses_Vector

def computeKmeansResult(wfVector, courseINF, num=5):
    words_Frequency_Vector = scale(np.array(wfVector))
    kmeans = KMeans(n_clusters=num)
    kmeans.fit(words_Frequency_Vector)

    silhouette_avg = metrics.silhouette_score(words_Frequency_Vector, kmeans.labels_)
    print(silhouette_avg)

    courseCluster = defaultdict(list)
    for index, label in enumerate(kmeans.labels_):
        courseCluster[label].append(courseINF[index])
    return {str(key):value for key, value in courseCluster.items()}

def saveKmeansResult(cluster_Result):
    fileName = 'Kmeans_Result/{}_courses_cluster.json'.format(jsonFileName)
    with open(fileName, 'w', encoding='utf-8') as f:
        f.write(json.dumps(cluster_Result, indent=4, ensure_ascii=False))
    
courseINF = loadCourseInformationFromFile()

course = getCourseInformation(courseINF)
words = decodeWords(course)
jiebaWords = getJiebaWords(words)
wordsSet = [w for e in jiebaWords for w in e]
course_KeyWords = getCourseKeyWords(set(wordsSet))


all_courseWords = getCourseTraining_Words(courseINF)
saveCourseWords(course_KeyWords)

courses_Vector = createCourseVector(all_courseWords)
cluster_Result = computeKmeansResult(courses_Vector, courseINF, 10)
saveKmeansResult(cluster_Result)
