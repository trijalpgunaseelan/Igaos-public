NAME          HAVERLY1
ROWS
 N  COST
 E  poolbal
 E  poolqual
 L  demX
 L  demY
 L  specX
 L  specY
COLUMNS
    A  COST  6
    A  poolbal  1
    A  poolqual  -3
    B  COST  16
    B  poolbal  1
    B  poolqual  -1
    Cx  COST  1
    Cx  demX  1
    Cx  specX  -0.5
    Cy  COST  -5
    Cy  demY  1
    Cy  specY  0.5
    Px  COST  -9
    Px  poolbal  -1
    Px  demX  1
    Px  specX  -2.5
    Py  COST  -15
    Py  poolbal  -1
    Py  demY  1
    Py  specY  -1.5
RHS
    RHS  poolbal  0
    RHS  poolqual  0
    RHS  demX  100
    RHS  demY  200
    RHS  specX  0
    RHS  specY  0
RANGES
BOUNDS
 UP BND  A  1000
 UP BND  B  1000
 UP BND  Cx  100
 UP BND  Cy  200
 UP BND  Px  100
 UP BND  Py  200
 LO BND  poolS  1
 UP BND  poolS  3
QCMATRIX  poolqual
    poolS  Px  0.5
    Px  poolS  0.5
    poolS  Py  0.5
    Py  poolS  0.5
QCMATRIX  specX
    poolS  Px  0.5
    Px  poolS  0.5
QCMATRIX  specY
    poolS  Py  0.5
    Py  poolS  0.5
ENDATA
