from flask import Flask, request
from flask import send_file

def create_app() -> Flask:
    app = Flask(__name__)

    @app.route('/tree_image', methods=['GET'])
    def add_image():
        return send_file('tree.png', mimetype='image/png') 
        
    @app.route('/mission', methods=['GET'])
    def add_image():
        return send_file('tree.png', mimetype='image/png')  

    return app


app = create_app()

if __name__ == '__main__':
    app.run(debug=True)
