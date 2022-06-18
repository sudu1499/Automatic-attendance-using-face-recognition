from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider

def cassandra_connect(): 
   cloud_config= {
        'secure_connect_bundle': 'secure-connect-registration.zip'
   }
   auth_provider = PlainTextAuthProvider('gxJOUTeUTqNpwYWTUlCtPgfh','ScWjbTOb3yWk7OtN47GcH4KbcdzWuTZBESlvl.KPSxnnUw-apDZTkeym96eZN2ZDZYkRRSTd96+ifo1-Z8CXIrHs1,JANWSBuG7pzlb6D10_PC+C7_-WU1YCDSlZp_YM')
   cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
   session = cluster.connect('Stu')
   return session
 

