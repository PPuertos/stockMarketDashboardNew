#### ESTE ES EL RESULTADO FINAL EFECTIVO ###
#### ESTE ES EL RESULTADO FINAL EFECTIVO ###
#### ESTE ES EL RESULTADO FINAL EFECTIVO ###

from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
import dash
import optimizationModule as om
import pandas as pd

data = om.portfolioActuals()

# Dataframe with information of all stocks of USA
allStocksDF = pd.read_csv('src/assets/stocks_list.csv')

# SCREENERS
popularStocks, mostActiveSymbols, dayGainerSymbols, symbols, closePrices = om.screeners()

# Links to use in the index_string
contentForSearchBar = [f"{sym} -- {name}"  for sym, name in zip(allStocksDF['symbol'], allStocksDF['name']) if sym in symbols]
links = {i:f'https://finance.yahoo.com/quote/{i}/' for i in symbols}

def create_app():
    app = Dash(__name__, use_pages=True, external_stylesheets=['assets/style.css', dbc.themes.BOOTSTRAP])
    
    # Define el HTML base con el navbar y el script para cerrar el offcanvas
    # Define el HTML base con el navbar y el script para cerrar el offcanvas
    # Define el HTML base con el navbar y el script para cerrar el offcanvas

    app.index_string = f"""
    <!DOCTYPE html>
    <html>
        <head>
            {{%metas%}}
            <title>Francisco P. Stock Portfolio Analysis</title>
            {{%css%}}
            <link href="https://fonts.googleapis.com/css2?family=Manrope:wght@400;700&display=swap" rel="stylesheet">
            
            <style>
                #search-input{{
                    display: inline;
                    border: None;
                    background-color: transparent;
                    outline: none;
                    transition: .8s;
                    width: 0px;  /* Estilo inicial */
                    text-align: center;  /* Alinea el texto al centro */
                }}

                #search-input.search-style {{
                    width: 20vw;  /* Estilo cuando se aplica la clase */
                    border-bottom: 1px solid #000;  /* Borde inferior */
                }}

                /* Estilos para el dropdown */
                .dropdown-menu {{
                    display: none;
                    position: absolute;
                    background-color: white;
                    border: 1px solid #ccc;
                    z-index: 1;
                    max-height: 150px;
                    overflow-y: auto;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                    margin-top:50px;
                    min-width: 20vw;
                }}

                .dropdown-item {{
                    padding: 8px;
                    cursor: pointer;
                }}

                .dropdown-item:hover {{
                    background-color: #f1f1f1; /* Color al pasar el mouse */
                }}
                
                @keyframes scroll {{
                    to {{
                        transform: translateX(calc(-100% - 20px));
                    }}
                }}
                
                .stockTicker:hover {{
                    cursor: pointer;
                }}
                
                .stockTicker:hover ul {{
                    animation-play-state: paused !important;
                }}
                
                .mac-select {{
                    'background-color':rgb(250,250,250) !important;
                }}
        
                .mac-select:hover {{
                    background-color: rgb(230,231,231) !important;
                }}
                
                #compute_button:hover {{
                    background-color: rgb(230,231,231) !important;
                }}
                
                #compute_button {{
                    background-image: url('data:image/svg+xml,%3Csvg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 384 512"%3E%3Cpath fill="rgb(100,100,100)" d="M73 39c-14.8-9.1-33.4-9.4-48.5-.9S0 62.6 0 80L0 432c0 17.4 9.4 33.4 24.5 41.9s33.7 8.1 48.5-.9L361 297c14.3-8.7 23-24.2 23-41s-8.7-32.2-23-41L73 39z"%3E%3C/path%3E%3C/svg%3E');
                }}
            
            </style>
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
                            <li class="nav-item" id="search-bar-box">
                                <input id="search-input" type="text" placeholder="Search for a Stock"></input>
                                <button class="nav-link" id="search-button"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512" style="fill: black; width: 1em; height: 1em; margin-right: 0.5em;"><path d="M416 208c0 45.9-14.9 88.3-40 122.7L502.6 457.4c12.5 12.5 12.5 32.8 0 45.3s-32.8 12.5-45.3 0L330.7 376c-34.4 25.2-76.8 40-122.7 40C93.1 416 0 322.9 0 208S93.1 0 208 0S416 93.1 416 208zM208 352a144 144 0 1 0 0-288 144 144 0 1 0 0 288z"/></svg></button>
                                <div id="dropdown-menu" class="dropdown-menu"></div>
                            </li>
                            <li class="nav-item"><a class="nav-link" aria-current="page" href="/">HOME</a></li>
                            <li class="nav-item"><a class="nav-link" href="/page1">PORTFOLIO</a></li>
                            <li class="nav-item"><a class="nav-link" href="/analysis">ANALYSIS</a></li>
                            <li class="nav-item"><a class="nav-link" href="/page2">PREDICTIONS</a></li>
                        </ul>
                    </div>
                </div>
            </div>
        </nav>

        <div class="container" style="margin-top: 100px;">
            {{%app_entry%}}
        </div>
        {{%config%}}
        {{%scripts%}}
        {{%renderer%}}

        <!-- Bootstrap JS -->
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>

        <!-- Agrega un EventListener al botón para enviarlo a Dash -->
        <script>
        
            
            // Obtener los elementos del DOM
            var inputElement = document.getElementById('search-input');
            var buttonElement = document.getElementById('search-button');
            var dropdownMenu = document.getElementById('dropdown-menu');

            // Opciones de ejemplo para el dropdown
            var options = {contentForSearchBar}
            console.log(options);
            var links = {links}

            // Función para mostrar el dropdown basado en el input
            inputElement.addEventListener('input', function() {{
                var query = inputElement.value.toLowerCase();
                dropdownMenu.innerHTML = '';  // Limpiar opciones anteriores

                if (query) {{
                    options.forEach(function(option) {{
                        // Separamos símbolo y nombre
                        var parts = option.split('--');
                        var symbol = parts[0].trim();
                        var name = parts[1] ? parts[1].trim() : '';
                        
                        if (symbol.toLowerCase().includes(query), name.toLowerCase().includes(query)) {{
                            var item = document.createElement('div');
                            item.textContent = option;
                            item.className = 'dropdown-item';
                            item.onclick = function() {{
                                inputElement.value = '';  // Poner opción seleccionada en el input
                                dropdownMenu.innerHTML = '';  // Limpiar el dropdown
                                dropdownMenu.style.display = 'none';  // Ocultar el dropdown
                                
                            if (links[symbol]) {{
                                window.open(links[symbol], '_blank');  // Redirigir al enlace
                                
                            }} else {{
                                console.log('No se encontró un enlace para la opción seleccionada');
                            }}

                            }};
                            dropdownMenu.appendChild(item);
                        }}
                    }});
                }}
                dropdownMenu.style.display = dropdownMenu.childElementCount ? 'block' : 'none';  // Mostrar el dropdown si hay opciones
            }});

            // Función para alternar el estilo
            buttonElement.addEventListener('click', function(event) {{
                event.stopPropagation();  // Evita que el clic en el botón cierre el input
                inputElement.classList.toggle('search-style');  // Alterna la clase custom-style

                // Alternar el ancho directamente
                if (inputElement.classList.contains('search-style')) {{
                    inputElement.style.width = '20vw';  // Ampliar el input
                    inputElement.value = ''
                }} else {{
                    inputElement.value = ''
                    dropdownMenu.style.display = 'none';
                    inputElement.style.width = '0px';  // Revertir a ancho cero
                }}
            }});

            // Restablecer el estilo y el dropdown si se hace clic en otro lugar
            document.addEventListener('click', function(event) {{
                if (!inputElement.contains(event.target) && !buttonElement.contains(event.target)) {{
                    inputElement.classList.remove('search-style');  // Eliminar la clase
                    inputElement.style.width = '0px';  // Restablecer el ancho
                    dropdownMenu.innerHTML = '';  // Limpiar el dropdown
                    inputElement.value = ''
                    dropdownMenu.style.display = 'none';  // Ocultar el dropdown
                }}
            }});
            
        </script>
        </body>
    </html>
    """
    
    app.layout = html.Div([
        dash.page_container,
        dcc.Store(id='portfolio_main_data', data=data)
    ])
    server = app.server
    
    return app

if __name__ == "__main__":
    app = create_app()
    app.run_server(debug=False, port=8051)