import flask
import ruamel.yaml
import sassutils.wsgi


def create_app(conffile='./config.yml'):
    app = flask.Flask(__name__)

    app.wsgi_app = sassutils.wsgi.SassMiddleware(app.wsgi_app, {
        'leitmotiv.webui': (
            'static/scss',
            'static/css',
            '/static/css'
        )
    })

    from leitmotiv.webui.view import view
    app.register_blueprint(view)

    from leitmotiv import library
    with open(conffile) as f:
        yaml = ruamel.yaml.YAML()
        config = yaml.load(f)['leitmotiv']

    library.LIBRARY_PATH = config['library']['path']
    app.config['IMAGE_DISTANCES'] = config['webui']['distances']

    return app
