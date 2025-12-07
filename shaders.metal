#include <metal_stdlib>
using namespace metal;

struct Vertex {
    float2 position;
    float2 textureCoordinates;
};

struct VertexOut {
    float4 position [[position]];
    float2 textureCoordinates;
};

vertex VertexOut
vertexShader(uint vertexID [[vertex_id]],
             constant Vertex* vertexPositions)
{
    return VertexOut {float4(vertexPositions[vertexID].position, 0, 1), vertexPositions[vertexID].textureCoordinates};
}

fragment float4 fragmentShader(VertexOut vertexOutPositions [[stage_in]], constant float2& translate [[buffer(1)]], constant float2x2& rotation_matrix [[buffer(2)]], texture2d<float> colorTexture [[texture(0)]]) {
    constexpr sampler textureSampler (mag_filter::linear, min_filter::linear, s_address::repeat, t_address::repeat);
     vertexOutPositions.textureCoordinates = rotation_matrix * vertexOutPositions.textureCoordinates;
    vertexOutPositions.textureCoordinates += translate;

    //return colorTexture.sample(textureSampler, texcoord);
    return colorTexture.sample(textureSampler, vertexOutPositions.textureCoordinates);
}
