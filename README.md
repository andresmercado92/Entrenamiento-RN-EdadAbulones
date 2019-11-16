# Entrenamiento-RN-EdadAbulones
Trabajo final para la asignatura de Inteligencia Artificial. La actividad consiste en entrenar una red neuronal con un dataset público y mediante de un informe mostrar los resultados. 

<strong> La docente a cargo de esta asignatura es PhD. Ing. Idanis Diaz Bolaño </strong>
Los integrantes del trabajo final son:
Mauricio Rodríguez Díaz Cod. 2011114131
Carmen Daly Vega Pérez Cod.
Andrés Mercado Niño Cod.
Paula Hernandez Vazquez Cod.

# Dataset usado para el entrenamiento de la RN.
<li> https://archive.ics.uci.edu/ml/datasets/abalone </li>

Donante de la base de datos al repositorio
Sam Waugh (Sam.Waugh@cs.utas.edu.au)
Department of Computer Science, University of Tasmania

Con este conjunto de datos entrenaremos la red neuronal para la predicción de edad del abulón. (Más informacion en el enlace).


<h3>Informacion de los atributos del dataset: </h3>
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
