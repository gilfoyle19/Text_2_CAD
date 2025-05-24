"""
TO_DO: Last updated 19-05-2025
1. List down the assembly models that to be created using cadquery.
2. Go through the cadquery documentation and add the methods that are not included in the cheatsheet.
3. Read the LIMA paper and CeADAR ultimate guide to fine-tuning the LLMs.
4. Read the Harvard CS197 document.
5. STL to point cloud conversion function.
6. Figure out how to create a dataset (prompt and code pairs, or any other additional information).
6. Create a mini dataset using intermediate shapes and models and fine-tune the LLM.
7. Evaluate the performance of the fine-tuned LLM on the dataset.
8. Then proceed with complex shapes and models and assembly models.

"""




#This is the cadquery cheatsheet for the dataset generation for fine-tuning the LLM.
#The characteristics of the dataset are:
# 1. Structure: 1. Prompt to create a CAD model with its unique parameters: "Create a rectangle with {length} 50 mm and {breadth} 20mm". 
# 2. File name : rectangle.py 
# 3. Comment with step by step explanation of the code. 
# 4. A modular code that uses the parameters from the prompt and create the code. 
# 5. A line of code to export the model to STL format.
# 6. Create examples using constraints (Step 1 to 5) (Note: constraint relationships between the shapes should be clearly mentioned in the comments for LLM to reason better). 
# 7. Repeat the above steps for different shapes with random parameters to generate the dataset.

#Dataset validation: Normalize -> Quantize the parameters -> Simplify the CAD model -> validate the model (code & geometry) -> check for duplicates and remove them. 


#Dataset


"""
Reasoning About Geometry and Angles
-----------------------------------

When working with CadQuery, it's essential to reason step-by-step about geometry and angles, especially for LLMs that may not be proficient at visualizing or performing mathematical calculations. Here are some tips to help LLMs better understand and work with geometric concepts:

1. Break down complex shapes into simpler components: Identify the basic shapes that make up the object, such as rectangles, circles, triangles, or cylinders. This makes it easier to understand the overall structure and dimensions of the object.

2. Use cardinal directions and relative positions: Describe the position and orientation of shapes using cardinal directions (e.g., north, south, east, west) and relative positions (e.g., above, below, left, right). This helps establish a clear understanding of the spatial relationships between components.

3. Specify angles using degrees or radians: When describing angles, always specify whether the angle is in degrees or radians. This helps avoid confusion and ensures that the correct values are used in the CadQuery code.

4. Use trigonometric functions for angle calculations: If angle calculations are required, use trigonometric functions like sine, cosine, and tangent to determine the necessary dimensions. Provide step-by-step explanations of the calculations to help LLMs understand the process.

5. Employ symmetry and repetition: Identify patterns of symmetry and repetition in the object, as this can simplify the modeling process. For example, if an object has multiple identical components, you can create one component and then use transformations (e.g., translations, rotations) to position the copies.

6. Utilize workplanes and local coordinate systems: Use workplanes to establish local coordinate systems that make it easier to reason about the position and orientation of shapes. This is particularly helpful when creating complex objects with multiple components.

7. Sketch 2D profiles before extruding: When creating 3D shapes, it often helps to sketch the 2D profile first and then extrude it. This allows you to focus on the cross-sectional shape and dimensions before considering the depth or height of the object.

"""
"""
Best Practices for CadQuery Code:

1. Use descriptive variable names: Choose variable names that clearly describe the purpose of the variable, such as "length", "width", "height", or "radius". This makes the code more readable and easier to understand.
2. Modularize the code: Break the code into smaller, reusable functions that perform specific tasks. This makes it easier to maintain and update the code, as well as to understand the overall structure of the model.
3. Add comments and docstrings: Provide clear comments and docstrings to explain the purpose of each function and the parameters it takes. This helps others (and yourself) understand the code when revisiting it later.
4. Use consistent formatting: Follow a consistent coding style, such as PEP 8 for Python, to improve readability. This includes using proper indentation, spacing, and line breaks.
5. Test the code: Regularly test the code to ensure that it produces the desired output. This helps catch errors early and makes it easier to debug any issues that arise.
"""

METHODS = """

## 3D Construction
- box(length, width, height)
- sphere(radius)
- cylinder(height, radius)
- text(txt, fontsize, distance)
- extrude(until)
- revolve(angleDegrees)
- loft(ruled)
- sweep(path, isFrenet, transitionMode)
- cutBlind(until)
- cutThruAll()
- hole(diameter, depth)
- shell(thickness)
- fillet(radius)
- chamfer(length)
- union(shape)
- cut(shape)
- intersect(shape)

## 2D Construction
- rect(xLen, yLen)
- circle(radius)
- ellipse(x_radius, y_radius)
- center(x, y)
- moveTo(x, y)
- move(xDist, yDist)
- lineTo(x, y)
- line(xDist, yDist)
- polarLine(distance, angle)
- vLine(distance)
- hLine(distance)
- polyline(listOfXYTuple)

## Sketching
- rect(w, h)
- circle(r)
- ellipse(a1, a2)
- trapezoid(w, h, a1)
- regularPolygon(r, n)
- polygon(pts)
- fillet(d)
- chamfer(d)
- finalize()

## Export
- shape.val().exportStl(path)

## Selector String Modifiers
- | (Parallel to)
- # (Perpendicular to)
- +/- (Pos/Neg direction)
- \> (Max)
- < (Min)
- % (Curve/surface type)

## Selector Methods
- faces(selector)
- edges(selector)
- vertices(selector)
- solids(selector)
- shells(selector)

## Workplane Positioning
- translate(Vector(x, y, z))
- rotateAboutCenter(Vector(x, y, z), angleDegrees)
- rotate(Vector(x, y, z), Vector(x, 
y, z), angleDegrees)"""

"""
Examples Shapes:

Primitives: Box, Sphere, Cylinder, cone, Tirus, Wedge
Common Shapes: Rectangle, Circle, Ellipse, Polygon, Regular Polygon
Common models: 
Spur Gear
Helical Gear
Bevel Gear
Worm Gear
Rack and Pinion
Ball Bearing
Roller Bearing
Thrust Bearing
Shaft
Keyway
Pulley
Flange
Bolt
Nut
Washer
Screw (Machine)
Screw (Self-tapping)
Rivet
Spring (Compression)
Spring (Torsion)
Spring (Extension)
Cam
Crankshaft
Piston
Connecting Rod
Valve
Pipe Fitting (Elbow)
Pipe Fitting (Tee)
Pipe Fitting (Reducer)
Hydraulic Cylinder
Pneumatic Actuator
Wall Section
Door
Window
Roof Truss
Beam (I-Beam)
Beam (H-Beam)
Column
HVAC Duct
Plumbing Pipe
Phone Case
Bottle
Cap (Screw-on)
Cap (Snap-on)
Handle (Ergonomic)
Knob
Hinge
Latch
Bracket 

Assemnbly models: 
Need to be added...
"""

EXAMPLES = """

(Just a placeholder, format might change)
#1
Code: import cadquery as cq\n\n# parameters\nlength =4\nheight = 3\nthickness =
    0.5\nfillet_radius = 0.5\n\n# functions\nresult = cq.Workplane("XY" ).box(length,
    height, thickness).edges("|Z").fillet(fillet_radius)
Name of Part: Simple Rectangular Plate
label: A rectangular plate with length of 4 mm, height of 3 mm and a thickness of
    0.5 mm\nThe edges of the plate are filleted with a fillet radius of 0.5 mm


"""