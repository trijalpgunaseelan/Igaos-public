NAME          BLENDINFEAS
* A refinery blend that cannot be made.
* Three crudes, one gasoline pool.  The planner has asked for 1,000 bbl/day
* at an average sulfur of 0.40 wt%.  Only the sweet crude is below that spec,
* and there are at most 600 bbl/day of it.  The volume target and the sulfur
* specification are therefore incompatible -- but nothing in the model says
* which pair is at fault, which is exactly what an IIS is for.
ROWS
 N  COST
 G  DEMAND
 L  SULFUR
COLUMNS
    SWEET     COST            45.0   DEMAND           1.0
    SWEET     SULFUR           0.1
    MEDIUM    COST            38.0   DEMAND           1.0
    MEDIUM    SULFUR           0.8
    SOUR      COST            32.0   DEMAND           1.0
    SOUR      SULFUR           1.6
RHS
    RHS       DEMAND        1000.0   SULFUR           0.0
BOUNDS
 UP BND       SWEET          600.0
 UP BND       MEDIUM         800.0
 UP BND       SOUR           800.0
ENDATA
