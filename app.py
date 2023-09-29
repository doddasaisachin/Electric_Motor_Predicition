from flask import Flask,render_template,request
import pickle
application=Flask(__name__)
app=application

def scaling(model,a,b,c,d,e,f,g,h,i):
    scaled=model.transform([[a,b,c,d,e,f,g,h,i]])
    return scaled

def predict_motor_speed(model,array):
    pred=model.predict(array)
    return pred

@app.route('/',methods=['GET','POST'])
def home():
    if request.method=='GET':
        return render_template('index.html')
    else:
        ambient=request.form.get('ambient')
        coolant=request.form.get('coolant')
        u_d=request.form.get('u_d')
        u_q=request.form.get('u_q')
        torque=request.form.get('torque')
        i_d=request.form.get('i_d')
        i_q=request.form.get('i_q')
        pm=request.form.get('pm')
        stator_tooth=request.form.get('stator_tooth')

        scaler=pickle.load(open('scaler.pkl','rb'))
        reg_model=pickle.load(open('model.pkl','rb'))
        
        scaled_values=scaling(scaler,
              float(ambient),float(coolant),float(u_d),
              float(u_q),float(torque),float(i_d),
              float(i_q),float(pm), float(stator_tooth)
              )
        predicted_value=predict_motor_speed(reg_model,scaled_values)
        return render_template('index.html',result=predicted_value[0])
if __name__=='__main__':
    app.run(host='0.0.0.0')