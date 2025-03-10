# Entrenamiento-RN-EdadAbulones
Trabajo final para la asignatura de Inteligencia Artificial. La actividad consiste en entrenar una red neuronal con un conjunto de datos público y mediante de un informe mostrar los resultados. 

<strong> La docente a cargo de esta asignatura es PhD. Ing. Idanis Diaz Bolaño </strong>

Los integrantes del trabajo final son:
<table>
  <tr>
    <td>Mauricio Rodríguez Díaz</td><td> Cod. 2011114131</td>
  </tr>
  <tr>
    <td>Carmen Daly Vega Pérez</td><td> Cod.</td>
  </tr>
  <tr>
    <td>Andrés Mercado Niño</td><td> Cod.</td>
  </tr>
  <tr>
    <td>Paula Hernandez Vazquez</td><td> Cod.</td>
  </tr>
</table>

# Conjunto de datos usado para el entrenamiento de la RN.
El conjunto de datos puede encontrarse en el siguiente enlace:
<li> https://archive.ics.uci.edu/ml/datasets/abalone </li><br>

<strong>Donante de la base de datos al repositorio</strong>
<li> Sam Waugh (Sam.Waugh@cs.utas.edu.au)</li>
<li>Department of Computer Science, University of Tasmania</li><br>
Con este conjunto de datos entrenaremos la red neuronal para la predicción de edad del abulón. (Más informacion en el enlace).


<h3>Informacion de los atributos del conjunto de datos: </h3>
<p>Se proporciona el nombre del atributo, el tipo de atributo, la unidad de medida y una breve descripción. El número de anillos es el valor a predecir: ya sea como un valor continuo o como un problema de clasificación.</p>
<table>
  <tr>
    <td><strong>Nombre</strong></td>
    <td><strong>Tipo de datos</strong></td>
    <td><strong>Unidad de medida</strong></td>
    <td><strong>Descripción</strong></td>
  </tr>
  <tr>
    <td>Sexo</td>
    <td>nominal</td>
    <td>-<td>
    M, F y I (Infante)
  </tr>
  <tr>
    <td>Longitud</td>
    <td>continuo</td>
    <td>mm</td>
    <td>Medida de concha más larga</td>
  </tr>
  <tr>
    <td>Diámetro</td>
    <td>continuo</td>
    <td>mm</td>
    <td>perpendicular a la longitud</td>
  </tr>
  <tr>
  	<td>
      Altura
    </td>
    <td>
      continuo
    </td>
    <td>
      mm
    </td>
    <td>
      con carne en concha
    </td>
  </tr>
  <tr>
    <td>	
      Peso entero
    </td>
    <td>
      continuo
    </td>
    <td>
      gramos
    </td>
    <td>
      abulón entero
    </td>
  </tr>
  <tr>
    <td>
      Peso desvainado
    </td>
    <td>
      continuo
    </td>
    <td>
      gramos
    </td>
    <td>
      peso de la carne
    </td>
  </tr>
  <tr>
    <td>
      Peso de las vísceras
    </td>
    <td>
      continuo
    </td>
    <td>
      gramos
    </td>
    <td>
      peso intestinal (después del sangrado)
    </td>
  </tr>
  <tr>
    <td>
      Peso de la cáscara
    </td>
    <td>
      continuo
    </td>
    <td>
      gramos
    </td>
    <td>
      después del secado
    </td>
  </tr>
  <tr>
    <td>
      Anillos
    </td>
    <td>
       entero
    </td>
    <td>
      -
    </td>
    <td>
      +1.5 da la edad en años
    </td>
  </tr>
</table>
