import face_recognition as face 
import numpy as np 
import cv2
import time
import paho.mqtt.client as mqtt
import json

NETPIE_HOST = "mqtt.netpie.io"
CLIENT_ID = "0781870a-a856-40fa-a09c-eb8ff8a6b635"
DEVICE_TOKEN = "BrWxH2G5AN1aouM5UeD1ERY2zWhCTT2Q"

data_name = {'name': 0}

def on_connect(client, userdata, flags, rc):
    print ("Result from connect: {}".format(
        mqtt.connack_string(rc)))
    client.subscribe("@shadow/data/updated")
    
client = mqtt.Client(protocol=mqtt.MQTTv311,
                     client_id=CLIENT_ID, clean_session=True)
client.username_pw_set(DEVICE_TOKEN)
client.on_connect = on_connect
client.connect(NETPIE_HOST , 1883)
client.loop_start()

try:
    video_capture = cv2.VideoCapture(0) 

    name_image = face.load_image_file("name.jpg")
    name_face_encoding = face.face_encodings(name_image)[0]

    name2_image = face.load_image_file("name2.jpg")
    name2_face_encoding = face.face_encodings(name_image)[0]

    face_locations = []
    face_encodings = []
    face_names = []
    face_percent = []

    process_this_frame = True
    
    known_face_encodings = [name_face_encoding, name2_face_encoding]
    known_face_names = ["Ketsara Rawangkai", "Janthima Seesa"]


    while True:
    
        ret, frame = video_capture.read()
        if ret:
        
            small_frame = cv2.resize(frame, (0,0), fx=0.5,fy=0.5)
        
            rgb_small_frame = small_frame[:,:,::-1]

            face_names = []
            face_percent = []

            if process_this_frame:
            
                face_locations = face.face_locations(rgb_small_frame, model="cnn")
            
                face_encodings = face.face_encodings(rgb_small_frame, face_locations)
            
            
                for face_encoding in face_encodings:
                    face_distances = face.face_distance(known_face_encodings, face_encoding)
                    best = np.argmin(face_distances)
                    face_percent_value = 1-face_distances[best]

                
                    if face_percent_value >= 0.5:
                        name = known_face_names[best]
                        percent = round(face_percent_value*100,2)
                        face_percent.append(percent)
                        print(name , ":" , "MATCH: "+str(percent)+"%", ":", time.strftime("%a, %d %b %Y %H:%M:%S"))
                        data_name['name'] = name , time.strftime("%a, %d %b %Y %H:%M:%S")
                        print(json.dumps({ "data" : data_name }))
                        client.publish('@shadow/data/update', json.dumps({"data" : data_name }), 1)
                        time.sleep(5)
                    else:
                        name = "UNKNOWN"
                        face_percent.append(0)
                        print(name)
                        data_name['name'] = name , time.strftime("%a, %d %b %Y %H:%M:%S")
                        print(json.dumps({ "data" : data_name }))
                        client.publish('@shadow/data/update', json.dumps({"data" : data_name }), 1)
                        time.sleep(5)
                    face_names.append(name)

        
            for (top,right,bottom, left), name, percent in zip(face_locations, face_names, face_percent):
                top*= 2
                right*= 2
                bottom*= 2
                left*= 2

                if name == "UNKNOWN":
                    color = [46,2,209]
                else:
                    color = [255,102,51]

                cv2.rectangle(frame, (left,top), (right,bottom), color, 2)
                cv2.rectangle(frame, (left-1, top -30), (right+1,top), color, cv2.FILLED)
                cv2.rectangle(frame, (left-1, bottom), (right+1,bottom+30), color, cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left+6, top-6), font, 0.6, (255,255,255), 1)
                cv2.putText(frame, "MATCH: "+str(percent)+"%", (left+6, bottom+23), font, 0.6, (255,255,255), 1)


        
            process_this_frame = not process_this_frame

        
            cv2.imshow("Video", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        else:
            break


    video_capture.release()
    cv2.destroyAllWindows()
    
except KeyboardInterrupt:
    pass
client.loop_start()
client.disconnect()