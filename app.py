from flask import Flask, request, jsonify, render_template
import requests

import itchat_app
import modules

app = Flask(__name__)

backend_url = "http://127.0.0.1:5000"  # 后端接口地址

@app.route('/')
def index():
    welcome_message = "您好！接下来您可以在下面聊天框输入想问的问题哦！"
    return render_template('index.html', welcome_message=welcome_message)

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()
    user_input = data['message']

    # 将用户输入发送给后端接口
    response = requests.post(backend_url + '/answer', json={"question": user_input})
   # bot_response = response.json()['answer']
    bot_response = itchat_app.solve_input(user_input)
    print(bot_response)
    return jsonify({'message': bot_response})

if __name__ == '__main__':
    app.run(debug=True)
