import pytz
import psycopg2
import datetime
import operator
from datetime import datetime
from stemming.porter2 import stem
from string import digits
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
import operator
from sklearn.externals import joblib
from sklearn.preprocessing import normalize

utc = pytz.UTC
#cur_date = utc.localize(datetime.now())
localtz = pytz.timezone('Asia/Kolkata')
#cur_date = utc.localize(datetime.now())

def db_connect():  ####### connecting to database, returns a connection to database
        try:
            conn = psycopg2.connect("dbname='recommendationengine' user='postgres' host='localhost' password='password'")
            print("I am connected to pgsql database")
        except:
            print ("I am unable to connect to the database")
        return conn

def removeHTML(s):    # fucntion that removes HTML tages from a string/dcoument/text
        i=0
        temp="";
        while(i<len(s)):
                if(s[i] == '<'):
                        start = i
                        while(s[i] !='>'):
                                i = i+1
                        i=i+1
                else:
                        temp = temp + s[i]
                        i = i+1
        return temp
#extraChar = ["{","}","[","]",'"',":","(",")","\n",",","/","'",".","\\","~","`","!","@","#","%","^","&","*","-","_","+","=",";","?","<",">","|"]
def refineData(line,sw): # function that remove stopwords from text, does the stemming, replace non-ascii characters
        extraChar = ["{","}","[","]",'"',":","(",")","\n",",","/","'",".","\\","~","`","!","@","#","%","^","&","*","-","_","+","=",";","?","<",">","|"]
        line = line.lower()
        for ec in extraChar:
                line =line.replace(ec," ")
        temp = ""
        for w in line.split():
                w = w.translate(None,digits)
                w = ''.join([i if ord(i) < 128 else ' ' for i in w])
                if(w == "y's"):
                        w = "y"
                w = stem(w)
                if(w not in sw):
                        temp = temp + " "+w
        return temp

##### Euclidean distance between two challenge (most similar challenges will have least distance)
def distance(content_sim_dict,weight_vector,vect1, vect2,conn):
        cur = conn.cursor()
        id1=int(vect1[0]) #challenge1 id
        id2=int(vect2[0]) #challenge2 id
        cur.execute("select interests_id from challenges_challenge_interest where challenge_id="+str(id1)+" intersect select interests_id from challenges_challenge_interest where challenge_id="+str(id2)+";")
        count = len(cur.fetchall()) # no. of interests matching between challenge1 & challenge2
        text_sim= content_sim_dict[id1][id2]
        vect1=np.append(vect1,text_sim) # appending text similarity to one of challenge atribute and keeping same attribute of other challenge to zero
        vect2=np.append(vect2,0)
        dist= euclidean_distances(vect1*weight_vector,vect2*weight_vector)  # Euclidian distance b/w two vectors
        return dist[0][0]+count*5 # adding weighted similarity of interests of challenges

def fetch_data(conn): ### fetch challenge_data from the database
        cur = conn.cursor()
        cur_date = utc.localize(datetime.now())
        cur.execute("select id, description from challenges_challenge;")
        temp  = cur.fetchall()    # temp array of challenge details
        temp_chal=np.asarray(temp)

        cur.execute("select challenge_id,interests_id from challenges_challenge_interest;")
        tag= cur.fetchall()  # array of challlenge interests

        cur.execute("select id,greycoins,created,deadline,company_id,man_hour from challenges_challenge;")
        vect=cur.fetchall()
#       vect = [[x[0],x[1],x[2],x[3],x[4],x[5]] for x in cur.fetchall()]
        vector=np.array(vect) # array contaning [challenge_id, greycoins, created, deadline,comapny_id, man_hour]

        chal_ids =((np.array(temp_chal))[:,0]).tolist() # list of challenge ids

        vector[:,[2]] = cur_date-vector[:,[2]] # vector[created] is replaced by (current date - created date)
        vector[:,[3]] = vector[:,[3]]-cur_date # vector[deadline] is replaced by (deadline - current date)
        for x in range(len(vector)): # converted above difference into seconds
                vector[x,2] = vector[x,2].total_seconds()
                vector[x,3] = vector[x,3].total_seconds()
        lst=[i for i in range(1,len(vector[0]))]
        vector[:,lst]=normalize(vector[:,lst],norm='l1',axis=0)  #normalization of each column except id's column (sum of each normalized column equal to 1)
        ret=[temp_chal[:,0],temp_chal[:,1],vector] # returns a list of challenge_id, challenge_description, array(id,greycoins,created,deadline,company_id,man_hour)
        return ret



def need_update(last_update,conn): #check if database need to be updated (Time complexity: O(n), n is the number of challenges)
        cur = conn.cursor()
        update_model = False  # Does model neeed to be updated
        cur.execute("select updated from challenges_challenge")
        update_date = [x[0] for x in cur.fetchall()]
        model_run_file  = open(last_update,"r")   # model_run contains last updated date of similarity model
        last_model_run = datetime.strptime((model_run_file.readlines())[0].replace("\n",""), "%Y-%m-%d %H:%M:%S.%f")
        localtz = pytz.timezone('Asia/Kolkata')
        #date_aware = localtz.localize(last_model_run)
        for date in update_date:
        #       print date, localtz.localize(last_model_run), (date > localtz.localize(last_model_run))
        #       print (date -  utc.localize(last_model_run))
                if(date > utc.localize(last_model_run)): # compaing  challenge update date with last model updated date
                        update_model = True
                        break
        return update_model  # returns boolean

def model_update(model_run,stop_words,fetched_data,conn):  #update similarity model and store in similarity.pkl file and returns a similarity matrix (complexity:O(n*n))
        cur = conn.cursor()
        if(need_update(model_run,conn)):
#       if(1): if you run for the 1st time there wont be any model and so no model_run file so create a model first
                chal_ids = fetched_data[0] #challenge_ids array
                temp_desc= fetched_data[1] #challenge_description
                print "need to update"
                sf = open(stop_words,"r")
                stop_words = ['content']   # list of stopwords around 550 words
                for w in sf:
                        w = w.replace("\n","")
                        stop_words.append(w.lower())

                chal_des = []    # store chal_description after refining (remove HTML, stopwords and stemming)
                for i in range(len(chal_ids)):
                        temp = removeHTML(temp_desc[i])
                        temp = refineData(temp,stop_words)
                        chal_des.append([chal_ids[i],temp])

                np_chal = np.array(chal_des) # array of challlenge description

        ##################### conversion of challenge description into tf-idf vector form ####################
                count_vect = CountVectorizer()
                train_counts = count_vect.fit_transform(np_chal[:,1])
                tfidf_transformer = TfidfTransformer()
                train_tfidf = tfidf_transformer.fit_transform(train_counts)


        #########################################################################################
                ecl_sim = [] # [ M * M ] matrix that store euclidian_distance between one document and rest documents
                for i in range(len(chal_des)):
                        sim = euclidean_distances(train_tfidf[i],train_tfidf)
                        sim = sim[0].tolist()
                        ecl_sim.append(sim)

                content_sim_dict = {} # dictionary that store  euclidian_distance between one document and rest documents
                                # content_sim_dict[chal_id] = {chal_idI:euclidean_distance(chal_id,chal_idI),..........} where chal_idI runs over all  challeneges

                for i in range(len(ecl_sim)):   #complexity: O(n*n), n=no. of challenges
                        simI = ecl_sim[i]
                        for j in range(len(simI)):
                                content_sim_dict.setdefault(int(chal_ids[i]),{})
                                content_sim_dict[int(chal_ids[i])][int(chal_ids[j])] = simI[j]

                vector=fetched_data[2]  ##id,greycoins,created,deadline,company_id,man_hour, text similarity
                weight_vector=np.array([0,15,7,7,25,5,41]) # weight of id,greycoins,created,deadline,company_id,man_hour, text similarity
                #### distance_matrix is a dictionary with values again a dictionary (distance_matrix[id1][id2] will give distance between two challenges with id1 and id2)
                distance_matrix={}  # complexity: O(m*n), n=no. of challenges, m= no. of challenge attributes
                for i in range(len(vector)): # storing similarity in distance_matrix dictionary
                        rowi=vector[i]
                        for j in range(len(vector)):
                                rowj=vector[j]
                                distance_matrix.setdefault(int(rowi[0]),{})
                                distance_matrix[int(rowi[0])][int(rowj[0])]=distance(content_sim_dict,weight_vector,rowi,rowj,conn)

                joblib.dump(distance_matrix,"similarity_model.pkl")    # storing the distance_matrix which our similarity model into a file
                model_update = open(model_run,"w")
                cur_timestamp = str((datetime.now()))
                model_update.write(cur_timestamp ) # updating last modified date of model
                return distance_matrix
        #       print "Updated"
        else:
                distance_matrix = joblib.load("similarity_model.pkl")  # loading model from similarity_model.pkl
                return distance_matrix
        #       print "No Need to Update"


def recommend(similarity_model, fetched_data,conn): #calculate recommendation for each user and stores in users_userprofile[recommendation]
        chal_ids = fetched_data[0] # challenge_ids
        vector=fetched_data[2]  ##id,greycoins,created,deadline,company_id,man_hour, text similarity
        cur = conn.cursor()
        cur.execute("select distinct  user_id from  challenges_activity  where activity_id<6 or  activity_id>10 ;")
        temp_act_users = cur.fetchall()
        act_users = [x[0] for x in temp_act_users] # users that have user activity

        act_wet={ # weighting of the user activity
                        11:0.285,       # solution created
                        12:0.285,       # solution edited
                        1:0.142,        # viewed
                        2:0.238,        # saved
                        3:0.190,        # undo saved
                        4:0,            # rejected
                        5:0.095         # undo rejected
                }

        temp_dist_mat = similarity_model # stored description similarity into a temp variable
        cur.execute("select id from auth_user")
        temp_user_ids = cur.fetchall()
        user_ids = [x[0] for x in temp_user_ids] # user ids

        for user in user_ids:   # complexity: O(n*n), no. of challenges
                cur.execute("select last_login,recommendation_date from auth_user,users_userprofile where auth_user.id=users_userprofile.user_id and user_id="+ str(user)+";")
                login=cur.fetchall() # list[last_login, recommendation_date] for a user
                should_rec = (len(login)>0)and ((login[0][1]==None) or ((login[0][0]!=None) and ( login[0][0] > login[0][1]))) #if either user is never recommended or last_login date of user > recommendation date
                if(should_rec):
                        print "login by : "+ str(user)
                        score = {} # recommendation score
                        if(user in act_users): # if the user has user activity
                                cur.execute("select target_id,activity_id  from  challenges_activity  where user_id="+ str(user)+"and ( activity_id<6 or  activity_id>10) ;")
                                user_activity  = cur.fetchall() # list of user activity [challenge_id, activity_id]

                                for chal_id,act_id in user_activity: # iterating over each challenge in activity and finding similar challenge
                                        sim_chal = temp_dist_mat[chal_id] # dictionary of challenge similar to chal_id
                                        if(chal_id in sim_chal):                # delete the same challenge from similarity dictionary
                                                del sim_chal[chal_id]
                                        x = len(chal_ids)/20            # we will consider around top 20 challenges which higher similarity
                                        threshold = (sim_chal[max(sim_chal,key=sim_chal.get)] + sim_chal[min(sim_chal,key=sim_chal.get)] )/float(x)  # threshold level of similarity we will consider challenges below this threshold(euclidean distance)

                                        for chal in sim_chal:   # iterating over similar challenges
                                                if(sim_chal[chal]<threshold):
                                                        if(chal in score):
                                                                score[int(chal)] = max( score[chal],(act_wet[act_id] *(1/ (sim_chal[chal]+1) ))) # take best score
                                                        else:
                                                                score[int(chal)] =  act_wet[act_id] *(1/ (sim_chal[chal]+1) )

                        else:   # user has no activity
                                for chal in chal_ids:
                                        score[int(chal)] = 1

                        for Idd in score.keys(): #updating scores of challenges, iterating over whole list of challenges
                                Id = int(Idd)
                                ls=np.where(vector[:,0]==Id)[0][0] #challenge_vector with given challenge_id
                                cur.execute("select interests_id from users_userprofile, users_userprofile_interests where user_id="+str(user)+" and users_userprofile.id=users_userprofile_interests.userprofile_id intersect select interests_id from challenges_challenge_interest where challenge_id="+str(Id)+";")
                                tags_matching =len(cur.fetchall())  # no. of matching interests of a user and a challenge
                                score[Id]=score[Id]*vector[ls][2]*vector[ls][3]+tags_matching # updating score
                        sorted_score = sorted(score.items(), key=operator.itemgetter(1), reverse=True)# sorted scores
                        recChal = [x[0] for x in sorted_score]  # list of recommended challenges in sorted order of their recommendation scor
                        recStr = "" # comma seperated string of recommended challenge_ids
                        for i in range(0,len(recChal)):
                                recStr = recStr+str(recChal[i])+","
                        temp_time =  str(utc.localize((datetime.now())))
#                       print "update recommendation: " + temp_time +" ,UTC:  " + str(datetime.now())
                        query = "update  users_userprofile set(recommendation,recommendation_date) = ( '"+ recStr[:-1] +"', '"+temp_time +"') where user_id="+ str(user) +";"
                        cur.execute(query)
        conn.commit()




######################################################################### recommedation for a user ##############################################
def rec_per_user(user_id, similarity_model, fetched_data,conn):  #order O(n*m) , n=no. of challenges in user activity, m=no. of total challenges
        chal_ids = fetched_data[0] # challenge_ids
        vector=fetched_data[2]  ##id,greycoins,created,deadline,company_id,man_hour, text similarity
        cur = conn.cursor()
        act_wet={ # weighting of the user activity
                        11:0.285,       # solution created
                        12:0.285,       # solution edited
                        1:0.142,        # viewed
                        2:0.238,        # saved
                        3:0.190,        # undo saved
                        4:0,            # rejected
                        5:0.095         # undo rejected
                }

        temp_dist_mat = similarity_model # stored description similarity into a temp variable

#       cur.execute("select last_login,recommendation_date from auth_user,users_userprofile where auth_user.id=users_userprofile.user_id and user_id="+ str(user_id)+";")
#       login=cur.fetchall() # list[last_login, recommendation_date] for a user
#        should_rec = (len(login)>0)and ((login[0][1]==None) or ((login[0][0]!=None) and ( login[0][0] > login[0][1]))) #if either user is never recommended or last_login date of user > recommendation date
#       if(should_rec):
        if(1):  #order O(n*m) , n=no. of challenges in user activity, m=no. of total challenges
#               print "login by : "+ str(user_id)
                score = {} # recommendation score
                cur.execute("select target_id,activity_id  from  challenges_activity  where user_id="+ str(user_id)+"and ( activity_id<6 or  activity_id>10) ;")
                user_activity  = cur.fetchall() # list of user activity [challenge_id, activity_id]

                if(len(user_activity)>0): # if the user has user activity
                        for chal_id,act_id in user_activity: # iterating over each challenge in activity and finding similar challenge
                                sim_chal = temp_dist_mat[chal_id] # dictionary of challenge similar to chal_id
                                if(chal_id in sim_chal):                # delete the same challenge from similarity dictionary
                                        del sim_chal[chal_id]
                                x = len(chal_ids)/20            # we will consider around top 20 challenges which higher similarity
                                threshold = (sim_chal[max(sim_chal,key=sim_chal.get)] + sim_chal[min(sim_chal,key=sim_chal.get)] )/float(x)  # threshold level of similarity we will consider challenges below this threshold(euclidean distance)

                                for chal in sim_chal:   # iterating over similar challenges
                                        if(sim_chal[chal]<threshold):
                                                if(chal in score):
                                                        score[int(chal)] = max( score[chal],(act_wet[act_id] *(1/ (sim_chal[chal]+1) ))) # take best score
                                                else:
                                                        score[int(chal)] =  act_wet[act_id] *(1/ (sim_chal[chal]+1) )
                else:   # user has no activity
                        for chal in chal_ids:
                               score[int(chal)] = 1

                for Idd in score.keys(): #updating scores of challenges, iterating over whole list of challenges
                        Id = int(Idd)
                        ls=np.where(vector[:,0]==Id)[0][0] #challenge_vector with given challenge_id
                        cur.execute("select interests_id from users_userprofile, users_userprofile_interests where user_id="+str(user_id)+" and users_userprofile.id=users_userprofile_interests.userprofile_id intersect select interests_id from challenges_challenge_interest where challenge_id="+str(Id)+";")
                        tags_matching =len(cur.fetchall())  # no. of matching interests of a user and a challenge
                        score[Id]=score[Id]*vector[ls][2]*vector[ls][3]+tags_matching # updating score
                sorted_score = sorted(score.items(), key=operator.itemgetter(1), reverse=True)# sorted scores
                recChal = [x[0] for x in sorted_score]  # list of recommended challenges in sorted order of their recommendation scor
                recStr = "" # comma seperated string of recommended challenge_ids
                for i in range(0,len(recChal)):
                        recStr = recStr+str(recChal[i])+","
                temp_time =  str(utc.localize((datetime.now())))
#               print recStr
                print recChal
                return recChal
#                       print "update recommendation: " + temp_time +" ,UTC:  " + str(datetime.now())
#                        query = "update  users_userprofile set(recommendation,recommendation_date) = ( '"+ recStr[:-1] +"', '"+temp_time +"') where user_id="+ str(user) +";"
#                       cur.execute(query)
        conn.commit()
