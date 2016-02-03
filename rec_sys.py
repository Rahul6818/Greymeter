from  rec_sys_fun  import db_connect, fetch_data, model_update, recommend,rec_per_user

def main():
        conn = db_connect() # connection to database
        data = fetch_data(conn) # fetch challenge data from database
        similarity_model = model_update("model_run","stopwords",data,conn) # compute similarity model where arg are (file that contains the last_model_updatation_date,stopwords,data,conn)
#    recommend(similarity_model,data,conn) # store recommendation into database
        user_id = raw_input("User Id: ")
        #this will store recommendation on database neither update recommendation_date on database
        rec_per_user(user_id, similarity_model,data,conn)

if __name__ == '__main__':
   main()
