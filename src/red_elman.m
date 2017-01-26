%{
	Titulo: Red de Elman - Práctica de redes neuronales de predicción
	Autor: Daniel González Alonso
	Fecha: 14/1/2017
%}
clear;


%%%% ____ CONSTANTES ____ %%%%
global RUTA_FICHERO     = '../data/iberdrola_Nov15-Dic16.csv';
global FORMATO_FICHERO	= '%*s,%f,%*f,%*f,%*f,%*f,%*f';
global TAU        = 20;   % Numero de iteraciones anteriores para el calculo
global T_MAX			= 235;	% Maximo numero de iteraciones de t
global ALPHA      = 0.25; % Momento (entre 0 y 1)
global NI         = 1;    % Numero de neuronas de entrada
global N0         = 10;	  % Numero de neuronas en la capa oculta
global N1				  = 1;	  % Numero de neuronas en la capa de salida


%%%% ____ FUNCIONES ____ %%%%
function y = sigmoide(x)
	%{
		Funcion Sigmoide
		x				    valor de entrada
	%}

	y = 1 ./ (1 + e.^(-x));
end


function y = sigmoidePrima(x)
	%{
		Derivada de la funcion sigmoide
	
		x				    valor de entrada
	%}

	y = e.^(-x) ./ (1 + e.^(-x)).^2;
end


function u0 = calculaU0(x, y0_anterior, w0)
	%{
		Funcion encargada de calcular el el vector u0 de la capa oculta
		
		x				    la entrada en la iteración actual (t)
		y0_anterior	la salida de la capa oculta en la iteración anterior (t-1)
		w0			    los pesos actuales de la capa oculta
	%}
  global NI;
  global N0;

	u0 = zeros(1, N0);
	for (j = 1:N0)
  
    for (k = 1:NI)
      u0(j) += w0(j, k) * x(k);
    end
    for (k = NI+1:NI+N0)
      u0(j) += w0(j, k) * y0_anterior(k-NI);
    end;
    u0(j) += w0(j, NI+N0+1);
    
	end
end


function u1 = calculaU1(y0, w1)
	%{	
		Funcion encargada de calcular el valor u1 de la capa de
		salida
	
		y0			    salida de la capa oculta en la iteracion actual (t)
		w1			    pesos actuales de la capa de salida
	%}
  global N0;
  global N1;
  
  u1 = zeros(1, N1);
  for (i = 1:N1)
    
    for (j = 1:N0)
      u1(i) += w1(i, j) * y0(j);
    end
    u1(i) += w1(i, N0+1);
    
  end
end


function delta_w0 = calculaDeltaW0(
    x_actual, y1_anterior,
    u1_anterior, w1, u0_anterior,
    x_anterior, y0_anterior_2
  )
  %{
    Funcion encargada de calcular las variaciones que hay que aplicar a los
    pesos de la capa de oculta en la fase hacia atras
  %}
  global NI;
  global N0;
  global N1;
  global ALPHA;
  
  delta_w0 = zeros(N0, N0+2);
  for (i = 1:N1)
    delta1 = -(x_actual(i) - y1_anterior(i)) * sigmoidePrima(u1_anterior(i));
   
    for (j = 1:N0)
      delta0 = delta1 * w1(i, j) * sigmoidePrima(u0_anterior(j));
      
      for (k = 1:NI)
        delta_w0(j, k) = -ALPHA * delta0 * x_anterior(k);       % k = 1, 2, ..., NI
      end
      for (k = NI+1:NI+N0)
        delta_w0(j, k) = -ALPHA * delta0 * y0_anterior_2(k-NI); % k = NI+1, ..., NI+N0
      end
      delta_w0(j, NI+N0+1) = -ALPHA * delta0;                   % k = NI+N0+1
    end
  end
end


function delta_w1 = calculaDeltaW1(x_actual, y1_anterior, u1_anterior, y0_anterior)
  %{
    Funcion encargada de calcular las variaciones que hay que aplicar a los
    pesos de la capa de salida en la fase hacia atras
  %}
  global N1;
  global N0;
  global ALPHA;

  delta_w1 = zeros(N1, N0+1);
  for (i = 1:N1)  
    delta1 = -(x_actual(i) - y1_anterior(i)) * sigmoidePrima(u1_anterior(i));
    
    for (j = 1:N0)
      delta_w1(i, j) = -ALPHA * delta1 * y0_anterior(j);
    end
    delta_w1(i, N0+1) = -ALPHA * delta1;
  end  
end



%%%% ____ INICIO ____ %%%%
% Obtenemos los datos de cierre del fichero
input = textread(RUTA_FICHERO, FORMATO_FICHERO, 'headerlines', 1)';

% Hold Out (2/3 - 1/3)
entrenamiento = input;
prueba = zeros(1,size(input,2)/3);
for (i = 1:size(input,2)/3)
  r = randi([1, size(entrenamiento,2)]);
  prueba(i) = entrenamiento(r);
  entrenamiento = entrenamiento([1:r-1 r+1:size(entrenamiento,2)]);
end;

% Creamos las matrices y vectores necesarios para la red neuronal
w0 = rand(N0, N0+NI+1) - 0.5 * ones(N0, N0+NI+1);
w1 = rand(N1, N0+1) - 0.5 * ones(N1, N0+1);
x_anterior = rand(1, NI) - 0.5 * ones(1, NI);
y0_anterior_2 = rand(1, N0) - 0.5 * ones(1, N0);
u0_anterior = rand(1, N0) - 0.5 * ones(1, N0);
y0_anterior = sigmoide(u0_anterior);
u1_anterior = rand(1, N1) - 0.5 * ones(1, N1);
y1_anterior = sigmoide(u1_anterior);


%%%% ____ APRENDIZAJE ____ %%%%
% Iteramos a lo largo de toda la entrada calculando las salidas de cada capa
% de nueronas
for (t = 1:size(entrenamiento,2))

	% ____ FASE HACIA ADELANTE ____
	% Obtenemos la entrada actual
	x = input(t);
	
  % Calculamos la salida de la capa oculta
	u0 = calculaU0(x, y0_anterior, w0);
  y0 = sigmoide(u0);
  
	% Calculamos la salida de la capa de salida
	u1 = calculaU1(y0, w1);
  y1 = u1;
  
  % ____ FASE HACIA ATRAS ____
  delta_w0 = calculaDeltaW0(x, y1_anterior, u1_anterior, w1, u0_anterior, x_anterior, y0_anterior_2);
  delta_w1 = calculaDeltaW1(x, y1_anterior, u1_anterior, y0_anterior);

  w0 += delta_w0;
  w1 += delta_w1;
  
  % Guardamos los datos actuales para la siguiente iteracion
  u0_anterior = u0;
  y0_anterior_2 = y0_anterior;
	y0_anterior = y0;
  x_anterior = x;
  u1_anterior = u1;
  y1_anterior = y1;
  
end


%%%% ____ TEST ____ %%%%
aciertos = 0;
for (t = 1:size(prueba,2))

	% Obtenemos la entrada actual
	x = prueba(t);
	
  % Calculamos la salida de la capa oculta
	u0 = calculaU0(x, y0_anterior, w0);
  y0 = sigmoide(u0);  

	% Calculamos la salida de la capa de salida
	u1 = calculaU1(y0, w1);
  y1 = u1;
  
  if (1/2 * (x - y1_anterior)^2 < 0.05)
    aciertos++;
  end
  
end

tasa_aciertos = aciertos / size(prueba,2)
