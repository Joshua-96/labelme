from OpenGL.GL import *  # noqa
from OpenGL.GL import shaders  # noqa
try:
    from OpenGL import NullFunctionError
except ImportError:
    from OpenGL.error import NullFunctionError
import numpy as np
import re

## For centralizing and managing vertex/fragment shader programs.

# shaders.GL.glEnable(shaders.GL.GL_AUTO_NORMAL)
# shaders.GL.glEnable(shaders.GL.GL_CULL_FACE)
def initShaders():
    global Shaders
    Shaders = [
        ShaderProgram(None, []),
        
        ## increases fragment alpha as the normal turns orthogonal to the view
        ## this is useful for viewing shells that enclose a volume (such as isosurfaces)
        ShaderProgram('balloon', [
            VertexShader("""
                varying vec3 normal;
                void main() {
                    // compute here for use in fragment shader
                    normal = normalize(gl_NormalMatrix * gl_Normal);
                    gl_FrontColor = gl_Color;
                    gl_BackColor = gl_Color;
                    gl_Position = ftransform();
                }
            """),
            FragmentShader("""
                varying vec3 normal;
                void main() {
                    vec4 color = gl_Color;
                    color.w = min(color.w + 2.0 * color.w * pow(normal.x*normal.x + normal.y*normal.y, 5.0), 1.0);
                    gl_FragColor = color;
                }
            """)
        ]),
        
        ## colors fragments based on face normals relative to view
        ## This means that the colors will change depending on how the view is rotated
        ShaderProgram('viewNormalColor', [   
            VertexShader("""
                varying vec3 normal;
                void main() {
                    // compute here for use in fragment shader
                    normal = normalize(gl_NormalMatrix * gl_Normal);
                    gl_FrontColor = gl_Color;
                    gl_BackColor = gl_Color;
                    gl_Position = ftransform();
                }
            """),
            FragmentShader("""
                varying vec3 normal;
                void main() {
                    vec4 color = gl_Color;
                    color.x = (normal.x + 1.0) * 0.5;
                    color.y = (normal.y + 1.0) * 0.5;
                    color.z = (normal.z + 1.0) * 0.5;
                    gl_FragColor = color;
                }
            """)
        ]),
        
        ## colors fragments based on absolute face normals.
        ShaderProgram('normalColor', [   
            VertexShader("""
                varying vec3 normal;
                void main() {
                    // compute here for use in fragment shader
                    normal = normalize(gl_Normal);
                    gl_FrontColor = gl_Color;
                    gl_BackColor = gl_Color;
                    gl_Position = ftransform();
                }
            """),
            FragmentShader("""
                varying vec3 normal;
                void main() {
                    vec4 color = gl_Color;
                    color.x = (normal.x + 1.0) * 0.5;
                    color.y = (normal.y + 1.0) * 0.5;
                    color.z = (normal.z + 1.0) * 0.5;
                    gl_FragColor = color;
                }
            """)
        ]),
        
        ## very simple simulation of lighting. 
        ## The light source position is always relative to the camera.
        ShaderProgram('shaded', [   
            VertexShader("""
                varying vec3 normal;
                void main() {
                    // compute here for use in fragment shader
                    normal = normalize(gl_NormalMatrix * gl_Normal);
                    gl_FrontColor = gl_Color;
                    gl_BackColor = gl_Color;
                    gl_Position = ftransform();
                }
            """),
            FragmentShader("""
                varying vec3 normal;
                void main() {
                    float p = dot(normal, normalize(vec3(1.0, -1.0, -1.0)));
                    p = p < 0. ? 0. : p * 0.8;
                    vec4 color = gl_Color;
                    color.x = color.x * (0.2 + p);
                    color.y = color.y * (0.2 + p);
                    color.z = color.z * (0.2 + p);
                    gl_FragColor = color;
                }
            """)
        ]),
        
        ## colors get brighter near edges of object
        ShaderProgram('edgeHilight', [   
            VertexShader("""
                varying vec3 normal;
                void main() {
                    // compute here for use in fragment shader
                    normal = normalize(gl_NormalMatrix * gl_Normal);
                    gl_FrontColor = gl_Color;
                    gl_BackColor = gl_Color;
                    gl_Position = ftransform();
                }
            """),
            FragmentShader("""
                varying vec3 normal;
                void main() {
                    vec4 color = gl_Color;
                    float s = pow(normal.x*normal.x + normal.y*normal.y, 2.0);
                    color.x = color.x + s * (1.0-color.x);
                    color.y = color.y + s * (1.0-color.y);
                    color.z = color.z + s * (1.0-color.z);
                    gl_FragColor = color;
                }
            """)
        ]),
        
        ## colors fragments by z-value.
        ## This is useful for coloring surface plots by height.
        ## This shader uses a uniform called "colorMap" to determine how to map the colors:
        ##    red   = pow(colorMap[0]*(z + colorMap[1]), colorMap[2])
        ##    green = pow(colorMap[3]*(z + colorMap[4]), colorMap[5])
        ##    blue  = pow(colorMap[6]*(z + colorMap[7]), colorMap[8])
        ## (set the values like this: shader['uniformMap'] = array([...])
        ShaderProgram('heightColor', [
            VertexShader("""
                varying vec4 pos;
                void main() {
                    gl_FrontColor = gl_Color;
                    gl_BackColor = gl_Color;
                    pos = gl_Vertex;
                    gl_Position = ftransform();
                }
            """),
            FragmentShader("""
                uniform float colorMap[9];
                varying vec4 pos;
                //out vec4 gl_FragColor;   // only needed for later glsl versions
                //in vec4 gl_Color;
                void main() {
                    vec4 color = gl_Color;
                    color.x = colorMap[0] * (pos.z + colorMap[1]);
                    if (colorMap[2] != 1.0)
                        color.x = pow(color.x, colorMap[2]);
                    color.x = color.x < 0. ? 0. : (color.x > 1. ? 1. : color.x);
                    
                    color.y = colorMap[3] * (pos.z + colorMap[4]);
                    if (colorMap[5] != 1.0)
                        color.y = pow(color.y, colorMap[5]);
                    color.y = color.y < 0. ? 0. : (color.y > 1. ? 1. : color.y);
                    
                    color.z = colorMap[6] * (pos.z + colorMap[7]);
                    if (colorMap[8] != 1.0)
                        color.z = pow(color.z, colorMap[8]);
                    color.z = color.z < 0. ? 0. : (color.z > 1. ? 1. : color.z);
                    
                    color.w = 1.0;
                    gl_FragColor = color;
                }
            """),
        ], uniforms={'colorMap': [1, 1, 1, 1, 0.5, 1, 1, 0, 1]}),
        ShaderProgram('pointSprite', [   ## allows specifying point size using normal.x
            ## See:
            ##
            ##  http://stackoverflow.com/questions/9609423/applying-part-of-a-texture-sprite-sheet-texture-map-to-a-point-sprite-in-ios
            ##  http://stackoverflow.com/questions/3497068/textured-points-in-opengl-es-2-0
            ##
            ##
            VertexShader("""
                void main() {
                    gl_FrontColor=gl_Color;
                    gl_PointSize = gl_Normal.x;
                    gl_Position = ftransform();
                } 
            """),
            #FragmentShader("""
                ##version 120
                #uniform sampler2D texture;
                #void main ( )
                #{
                    #gl_FragColor = texture2D(texture, gl_PointCoord) * gl_Color;
                #}
            #""")
        ]),
        ShaderProgram('shaded_gpu', [   
            VertexShader("""


                uniform mat4 u_Model;
                uniform mat4 u_View;
                uniform mat4 u_Projection;

                in vec3 a_Position;
                in vec3 a_Normal;
                in vec2 a_TextureCoordinates;

                out vec3 v_Position;
                out vec3 v_Normal;
                out vec2 v_TextureCoordinates;

                    out mat4 v_Model;
                    out mat4 v_View;
                    out mat4 v_Projection;

                    void main() {

                        v_Model = u_Model;
                        v_View = u_View;
                        v_Projection = u_Projection;

                        v_TextureCoordinates = a_TextureCoordinates;
                        v_Position = vec3(u_View * u_Model * vec4(a_Position, 1.0));
                        v_Normal = vec3(u_View * u_Model * vec4(a_Normal, 0.0));
                        gl_Position = u_Projection * u_View * u_Model * vec4(a_Position, 1.0);
		            }
            """),
            FragmentShader("""

                    in vec3 v_Position;
                    in vec3 v_Normal;
                    in vec2 v_TextureCoordinates;

                    in mat4 v_Model;
                    in mat4 v_View;
                    in mat4 v_Projection;

                    uniform sampler2D u_dem;
                    uniform sampler2D u_texture;

                    out vec4 color;

                    const float SIZE = 2.0;
                    const float RESOLUTION = 255.0;

                    const vec2 size = vec2(2.0 * SIZE / (RESOLUTION + 1.0), 0.0);
                    const ivec3 offset = ivec3(-1,0,1);

                    float getAltitude(vec4 pixel) {

                        float red = pixel.x;
                        float green = pixel.y;
                        float blue = pixel.z;
                        const float Z_SCALE = 5000.0 / SIZE;

			            return (red * 256.0 * 256.0 + green * 256.0 + blue) * 0.1 * 256.0 / Z_SCALE;
		                }

                    void main() {

                        float s11 = getAltitude(texture(u_dem, v_TextureCoordinates));
                        float s21 = getAltitude(textureOffset(u_dem, v_TextureCoordinates, offset.zy));
                        float s10 = getAltitude(textureOffset(u_dem, v_TextureCoordinates, offset.yx));

                        vec3 va = (vec3(size.xy, s21 - s11));
                        vec3 vb = (vec3(size.yx, s10 - s11));

                        vec3 normal = normalize(cross(va, vb));
                        //vec3 transformedNormal = normal;
                        vec3 transformedNormal = normalize(vec3(v_View * v_Model * vec4(normal, 0.0)));

                        vec3 lightVector = normalize(-v_Position);
                        float diffuse = max(dot(transformedNormal, lightVector), 0.1);

                        highp vec4 textureColor = texture(u_texture, v_TextureCoordinates);
                        color = vec4(textureColor.rgb * diffuse, textureColor.a);
                    }
            """)
        ]),
    ]


CompiledShaderPrograms = {}
    
def getShaderProgram(name):
    return ShaderProgram.names[name]

class Shader(object):
    def __init__(self, shaderType, code):
        self.shaderType = shaderType
        self.code = code
        self.compiled = None
        
    def shader(self):
        if self.compiled is None:
            try:
                self.compiled = shaders.compileShader(self.code, self.shaderType)
            except NullFunctionError:
                raise Exception("This OpenGL implementation does not support shader programs; many OpenGL features in pyqtgraph will not work.")
            except RuntimeError as exc:
                ## Format compile errors a bit more nicely
                if len(exc.args) == 3:
                    err, code, typ = exc.args
                    if not err.startswith('Shader compile failure'):
                        raise
                    code = code[0].decode('utf_8').split('\n')
                    err, c, msgs = err.partition(':')
                    err = err + '\n'
                    msgs = re.sub('b\'','',msgs)
                    msgs = re.sub('\'$','',msgs)
                    msgs = re.sub('\\\\n','\n',msgs)
                    msgs = msgs.split('\n')
                    errNums = [()] * len(code)
                    for i, msg in enumerate(msgs):
                        msg = msg.strip()
                        if msg == '':
                            continue
                        m = re.match(r'(\d+\:)?\d+\((\d+)\)', msg)
                        if m is not None:
                            line = int(m.groups()[1])
                            errNums[line-1] = errNums[line-1] + (str(i+1),)
                            #code[line-1] = '%d\t%s' % (i+1, code[line-1])
                        err = err + "%d %s\n" % (i+1, msg)
                    errNums = [','.join(n) for n in errNums]
                    maxlen = max(map(len, errNums))
                    code = [errNums[i] + " "*(maxlen-len(errNums[i])) + line for i, line in enumerate(code)]
                    err = err + '\n'.join(code)
                    raise Exception(err)
                else:
                    raise
        return self.compiled

class VertexShader(Shader):
    def __init__(self, code):
        Shader.__init__(self, GL_VERTEX_SHADER, code)
        
class FragmentShader(Shader):
    def __init__(self, code):
        Shader.__init__(self, GL_FRAGMENT_SHADER, code)
        
        
        

class ShaderProgram(object):
    names = {}
    
    def __init__(self, name, shaders, uniforms=None):
        self.name = name
        ShaderProgram.names[name] = self
        self.shaders = shaders
        self.prog = None
        self.blockData = {}
        self.uniformData = {}
        
        ## parse extra options from the shader definition
        if uniforms is not None:
            for k,v in uniforms.items():
                self[k] = v
        
    def setBlockData(self, blockName, data):
        if data is None:
            del self.blockData[blockName]
        else:
            self.blockData[blockName] = data

    def setUniformData(self, uniformName, data):
        if data is None:
            del self.uniformData[uniformName]
        else:
            self.uniformData[uniformName] = data
            
    def __setitem__(self, item, val):
        self.setUniformData(item, val)
        
    def __delitem__(self, item):
        self.setUniformData(item, None)

    def program(self):
        if self.prog is None:
            try:
                compiled = [s.shader() for s in self.shaders]  ## compile all shaders
                self.prog = shaders.compileProgram(*compiled)  ## compile program
            except:
                self.prog = -1
                raise
        return self.prog
        
    def __enter__(self):
        if len(self.shaders) > 0 and self.program() != -1:
            glUseProgram(self.program())
            
            try:
                ## load uniform values into program
                for uniformName, data in self.uniformData.items():
                    loc = self.uniform(uniformName)
                    if loc == -1:
                        raise Exception('Could not find uniform variable "%s"' % uniformName)
                    glUniform1fv(loc, len(data), np.array(data, dtype=np.float32))
                    
                ### bind buffer data to program blocks
                #if len(self.blockData) > 0:
                    #bindPoint = 1
                    #for blockName, data in self.blockData.items():
                        ### Program should have a uniform block declared:
                        ### 
                        ### layout (std140) uniform blockName {
                        ###     vec4 diffuse;
                        ### };
                        
                        ### pick any-old binding point. (there are a limited number of these per-program
                        #bindPoint = 1
                        
                        ### get the block index for a uniform variable in the shader
                        #blockIndex = glGetUniformBlockIndex(self.program(), blockName)
                        
                        ### give the shader block a binding point
                        #glUniformBlockBinding(self.program(), blockIndex, bindPoint)
                        
                        ### create a buffer
                        #buf = glGenBuffers(1)
                        #glBindBuffer(GL_UNIFORM_BUFFER, buf)
                        #glBufferData(GL_UNIFORM_BUFFER, size, data, GL_DYNAMIC_DRAW)
                        ### also possible to use glBufferSubData to fill parts of the buffer
                        
                        ### bind buffer to the same binding point
                        #glBindBufferBase(GL_UNIFORM_BUFFER, bindPoint, buf)
            except:
                glUseProgram(0)
                raise
                    
            
        
    def __exit__(self, *args):
        if len(self.shaders) > 0:
            glUseProgram(0)
        
    def uniform(self, name):
        """Return the location integer for a uniform variable in this program"""
        return glGetUniformLocation(self.program(), name.encode('utf_8'))

    #def uniformBlockInfo(self, blockName):
        #blockIndex = glGetUniformBlockIndex(self.program(), blockName)
        #count = glGetActiveUniformBlockiv(self.program(), blockIndex, GL_UNIFORM_BLOCK_ACTIVE_UNIFORMS)
        #indices = []
        #for i in range(count):
            #indices.append(glGetActiveUniformBlockiv(self.program(), blockIndex, GL_UNIFORM_BLOCK_ACTIVE_UNIFORM_INDICES))
        
class HeightColorShader(ShaderProgram):
    def __enter__(self):
        ## Program should have a uniform block declared:
        ## 
        ## layout (std140) uniform blockName {
        ##     vec4 diffuse;
        ##     vec4 ambient;
        ## };
        
        ## pick any-old binding point. (there are a limited number of these per-program
        bindPoint = 1
        
        ## get the block index for a uniform variable in the shader
        blockIndex = glGetUniformBlockIndex(self.program(), "blockName")
        
        ## give the shader block a binding point
        glUniformBlockBinding(self.program(), blockIndex, bindPoint)
        
        ## create a buffer
        buf = glGenBuffers(1)
        glBindBuffer(GL_UNIFORM_BUFFER, buf)
        glBufferData(GL_UNIFORM_BUFFER, size, data, GL_DYNAMIC_DRAW)
        ## also possible to use glBufferSubData to fill parts of the buffer
        
        ## bind buffer to the same binding point
        glBindBufferBase(GL_UNIFORM_BUFFER, bindPoint, buf)
        
initShaders()
