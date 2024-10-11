#### ESTE ES EL RESULTADO FINAL EFECTIVO ###
#### ESTE ES EL RESULTADO FINAL EFECTIVO ###
#### ESTE ES EL RESULTADO FINAL EFECTIVO ###

from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
import dash

def create_app():
    app = Dash(__name__, use_pages=True, external_stylesheets=['assets/style.css', dbc.themes.BOOTSTRAP])
    
    # Define el HTML base con el navbar y el script para cerrar el offcanvas
    # Define el HTML base con el navbar y el script para cerrar el offcanvas
    app.index_string = '''
    <!DOCTYPE html>
    <html>
        <head>
            {%metas%}
            <title>Francisco P. Stock Portfolio Analysis</title>
            {%css%}
            <!-- Agrega tus hojas de estilo aquí -->
            <link href="https://fonts.googleapis.com/css2?family=Manrope:wght@400;700&display=swap" rel="stylesheet">
        </head>
        <body>
            <nav class="navbar navbar-expand-md fixed-top bg-white">
                <div class="container-fluid">
                    <a class="navbar-brand" href="#">FRANCISCO PUERTOS</a>
                    <button class="navbar-toggler border-dark" type="button" data-bs-toggle="offcanvas" data-bs-target="#offcanvasNavbar" aria-controls="offcanvasNavbar" aria-label="Toggle navigation">
                        <span class="navbar-toggler-icon"></span>
                    </button>
                    <div class="offcanvas offcanvas-end bg-light" tabindex="-1" id="offcanvasNavbar" aria-labelledby="offcanvasNavbarLabel">
                        <div class="offcanvas-header">
                            <h5 class="offcanvas-title" id="offcanvasNavbarLabel">Portfolio</h5>
                            <button type="button" class="btn-close" data-bs-dismiss="offcanvas" aria-label="Close"></button>
                        </div>
                        <div class="offcanvas-body">
                            <ul class="navbar-nav justify-content-end flex-grow-1 pe-3">
                                <li class="nav-item"><a id='home_page' class="nav-link" aria-current="page" href="/">HOME</a></li>
                                <li class="nav-item"><a class="nav-link" href="/page1">PORTFOLIO</a></li>
                                <li class="nav-item"><a class="nav-link" href="#">ANALYSIS</a></li>
                                <li class="nav-item"><a class="nav-link" href="/page2">OPTIMIZATION</a></li>
                            </ul>
                        </div>
                    </div>
                </div>
            </nav>
            <div class="container" style="margin-top: 100px;">
                {%app_entry%}
            </div>
            {%config%}
            {%scripts%}
            {%renderer%}
            
            <!-- Bootstrap JS -->
            <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
            
            <!-- Script para cerrar el offcanvas al hacer clic en los enlaces -->
            <script>
                document.addEventListener('DOMContentLoaded', function () {
                    var offcanvasElement = document.getElementById('offcanvasNavbar');
                    var offcanvas = new bootstrap.Offcanvas(offcanvasElement);
                    var navLinks = document.querySelectorAll('.offcanvas-body .nav-link');
                    
                    navLinks.forEach(function (link) {
                        link.addEventListener('click', function () {
                            offcanvas.hide();  // Cierra el offcanvas cuando se hace clic en un enlace
                        });
                    });
                });
            </script>
        </body>
    </html>
    '''
    
layout = html.Div([
    html.Div("Hello there! This is the page of the stock, Welcome", className='h1'),
    dcc.Store(id='stock-page-store')
])