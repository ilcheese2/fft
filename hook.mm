#import <objc/runtime.h>
#import "Metal/Metal.h"
#import "QuartzCore/CAMetalLayer.h"
#include <CoreAudio/AudioHardware.h>
#include <CoreAudio/AudioHardwareTapping.h>
#include <CoreAudio/CoreAudio.h>
#include <mach/mach_time.h>
#include <Foundation/Foundation.h>
#include <CoreFoundation/CoreFoundation.h>
#include <dlfcn.h>
#include "butterflies.h"
#include <atomic>
#include <mutex>
#include <random>

struct Vertex {
    float position[2];
    float textureCoordinates[2];
};

struct FrameContext { // kill this guy
    int len;
    int offset;
    int oldestHistoryIndex;
    float sampleRate;
    float lastBeat;
    float aboveThreshold;
    Precomputed factors;
    float* buffer;
    float* imags;
    float* bandHistory;
};

float get_time() {
    mach_timebase_info_data_t timebase_info;
    mach_timebase_info(&timebase_info);
    uint64_t nanoseconds = mach_absolute_time() * timebase_info.numer / timebase_info.denom;
    return (float)nanoseconds / 1.0e9f;
}

std::mutex mutex;

class Renderer {
private:
    id<MTLTexture> framebuffer;
    id<MTLDevice> device;
    id<MTLLibrary> library;
    id<MTLCommandQueue> commandQueue;
    id<MTLBuffer> vertexBuffer;
    id<MTLRenderPipelineState> pipelineState;
    float last_frame;
    float progress;
    float currentTranslation[2];
    float pastTranslation[2];
    float currentRotation;
    float pastRotation;
    std::random_device rd;
    std::mt19937 mt{rd()};
    std::uniform_real_distribution<float> dist{0.0, 1.0};
    FrameContext* context;
    int bounces;
    
    void copy(id<CAMetalDrawable> drawable) {
        id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
        
        MTLRenderPassDescriptor* renderPassDescriptor = [MTLRenderPassDescriptor renderPassDescriptor];
        renderPassDescriptor.colorAttachments[0] = [MTLRenderPassColorAttachmentDescriptor new];
        renderPassDescriptor.colorAttachments[0].texture = drawable.texture;
        renderPassDescriptor.colorAttachments[0].loadAction = MTLLoadActionClear;
        renderPassDescriptor.colorAttachments[0].storeAction = MTLStoreActionStore;
        renderPassDescriptor.colorAttachments[0].clearColor = MTLClearColorMake(0.f, 0.f, 0.f, 1.0);
      
        id<MTLBlitCommandEncoder> blitCommandEncoder = [commandBuffer blitCommandEncoder];
        [blitCommandEncoder copyFromTexture:drawable.texture toTexture:framebuffer];
        [blitCommandEncoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
    }
public:
    Renderer(FrameContext* context) {
        device = MTLCreateSystemDefaultDevice();
        commandQueue = [device newCommandQueue];
        NSError *error = nil;
        library = [device newLibraryWithFile:@"/Users/ilcheese2/CLionProjects/untitled/cmake-build-debug/shaders.metallib" error:&error];
        Vertex squareVertices[] {
                {{-1, -1}, {0.0f, 1.0f}},
                {{-1,  1}, {0.0f, 0.0f}},
                {{ 1,  1}, {1.0f, 0.0f}},
                {{-1, -1}, {0.0f, 1.0f}},
                {{ 1,  1}, {1.0f, 0.0f}},
                {{ 1, -1}, {1.0f, 1.0f}}
        };
        vertexBuffer = [device newBufferWithBytes:&squareVertices length:6 * sizeof(Vertex) options:MTLResourceStorageModePrivate];
        this->context = context;
    }
    
    ~Renderer() {
        [device release];
        [framebuffer release];
        [library release];
        [vertexBuffer release];
    };
    
    void updateFramebuffer(id<CAMetalDrawable> drawable) {
        CGSize size = drawable.layer.drawableSize;
        if (framebuffer && size.width == framebuffer.width && size.height == framebuffer.height) {
            return;
        }
        if (framebuffer) [framebuffer release];
        MTLTextureDescriptor * texDesc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:drawable.layer.pixelFormat
                                                                                            width:size.width
                                                                                           height:size.height
                                                                                        mipmapped:NO];
        texDesc.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget;
        texDesc.storageMode = MTLStorageModePrivate;
        framebuffer = [device newTextureWithDescriptor:texDesc];
    };
    
    void draw(id<CAMetalDrawable> drawable) {
        @autoreleasepool {
            if (!pipelineState) {
                id<MTLFunction> vertexShader = [library newFunctionWithName:@"vertexShader"];
                id<MTLFunction> fragmentShader = [library newFunctionWithName:@"fragmentShader"];
                
                MTLRenderPipelineDescriptor* renderPipelineDescriptor = [MTLRenderPipelineDescriptor new];
                [renderPipelineDescriptor setLabel:@"pipeline"];
                [renderPipelineDescriptor setVertexFunction:vertexShader];
                [renderPipelineDescriptor setFragmentFunction:fragmentShader];
                
                [[renderPipelineDescriptor colorAttachments][0] setPixelFormat:drawable.layer.pixelFormat];
                
                NSError* error = nil;
                pipelineState = [device newRenderPipelineStateWithDescriptor:renderPipelineDescriptor error:&error];
                
                [renderPipelineDescriptor release];
            }
            
            float time = get_time();
            float delta_time = time - last_frame;
            last_frame = time;
            
            int maxBounces = 5;
            {
                std::lock_guard<std::mutex> lock(mutex);
                if (context->aboveThreshold > 0.0f && bounces == 0) {
                    bounces = maxBounces;
                    progress = 0.0f;
                }
            }
            
            if (!bounces) return;
            
            copy(drawable);
            
            id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
            
            MTLRenderPassDescriptor* renderPassDescriptor = [MTLRenderPassDescriptor renderPassDescriptor];
            renderPassDescriptor.colorAttachments[0] = [MTLRenderPassColorAttachmentDescriptor new];
            renderPassDescriptor.colorAttachments[0].texture = drawable.texture;
            renderPassDescriptor.colorAttachments[0].loadAction = MTLLoadActionClear;
            renderPassDescriptor.colorAttachments[0].storeAction = MTLStoreActionStore;
            renderPassDescriptor.colorAttachments[0].clearColor = MTLClearColorMake(0.f, 0.f, 0.f, 1.0);
            
            id<MTLRenderCommandEncoder> renderCommandEncoder = [commandBuffer renderCommandEncoderWithDescriptor:renderPassDescriptor];
            renderCommandEncoder.label = @"Forward pass";
            [renderCommandEncoder setCullMode:MTLCullModeBack];
            
            [renderCommandEncoder setRenderPipelineState:pipelineState];
            [renderCommandEncoder setVertexBuffer:vertexBuffer offset:0 atIndex:0];
            
            
            progress += delta_time * 30.0f;
            if (progress >= 1.0f) { // https://github.com/gasgiant/Camera-Shake/blob/master/Assets/GG%20Camera%20Shake/Runtime/BounceShake.cs
                progress = 0.0f;
                bounces--;
                pastRotation = currentRotation;
                pastTranslation[0] = currentTranslation[0];
                pastTranslation[1] = currentTranslation[1];
                
                float decay = 1 - ((float)bounces / (float)maxBounces);
                float rotationAmount = 0.04f * decay * decay;
                float displacementAmount = 0.02f * decay * decay;
                currentRotation = (dist(mt) - 0.5f) * rotationAmount;
                currentTranslation[0] = (dist(mt) - 0.5f) * displacementAmount;
                currentTranslation[1] = (dist(mt) - 0.5f) * displacementAmount;
            }
            
            float displacement[2];
            displacement[0] = pastTranslation[0] + (currentTranslation[0] - pastTranslation[0]) * progress;
            displacement[1] = pastTranslation[1] + (currentTranslation[1] - pastTranslation[1]) * progress;
            
            float angle = pastRotation + (currentRotation - pastRotation) * progress;
            float transformationMatrix[4];
            transformationMatrix[0] = cos(angle);
            transformationMatrix[1] = -sin(angle);
            transformationMatrix[2] = sin(angle);
            transformationMatrix[3] = cos(angle);
            
            [renderCommandEncoder setFragmentBytes:&displacement length:sizeof(float) * 2 atIndex:1];
            [renderCommandEncoder setFragmentBytes:&transformationMatrix length:sizeof(float) * 4 atIndex:2];
            
            
            [renderCommandEncoder setFragmentTexture:framebuffer atIndex:0];
            [renderCommandEncoder drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:6];
            
            [renderCommandEncoder endEncoding];
            [commandBuffer commit];
            [commandBuffer waitUntilCompleted];
        }
    };
};

typedef void (*RequestFuncType)(
    NSString*,
    CFDictionaryRef,
    void (^)(BOOL)
                                
);

typedef int (*RequestFuncType2)(
    NSString*
);

void requestPermission() {
    auto handle = dlopen("/System/Library/PrivateFrameworks/TCC.framework/Versions/A/TCC", RTLD_NOW);
    auto call = (RequestFuncType) dlsym(handle, "TCCAccessRequest");
    auto call2 = (RequestFuncType2) dlsym(handle, "TCCAccessPreflight");
    if (call2(@"kTCCServiceAudioCapture") == 0) return;
    call(@"kTCCServiceAudioCapture", nil, ^(BOOL success) {
        if (!success) {
            exit(1);
        }
    });
}

OSStatus callback(AudioObjectID inDevice, const AudioTimeStamp* inNow, const AudioBufferList* inInputData, const AudioTimeStamp* inInputTime, AudioBufferList* outOutputData, const AudioTimeStamp* inOutputTime, void* __nullable inClientData) {

    assert(inInputData->mNumberBuffers == 1);
    const AudioBuffer& buffer = inInputData->mBuffers[0];
    assert(buffer.mNumberChannels == 1);
    assert(buffer.mDataByteSize == 2048);

    FrameContext& context = *(FrameContext*)inClientData;
    
    memcpy(context.buffer + context.offset, buffer.mData, buffer.mDataByteSize);

    context.offset += buffer.mDataByteSize/4;
    if (context.offset >= context.len) {
        memset(context.imags, 0, context.len * sizeof(float));
        do_real_fft_precomputed(context.buffer, context.imags, context.factors, context.len);
        
        float bandLow = 20.0f;
        float bandHigh = 250.0f;

        int start = bandLow * (float)context.len/(float)context.sampleRate;
        int end_i = ceil(bandHigh * (float)context.len/(float)context.sampleRate);
        
        float energy = 0;
        
        for (int i = start; i <= end_i; i++) {
            energy += sqrt(context.buffer[i] * context.buffer[i] + context.imags[i] * context.imags[i]);
        }
        
        energy /= (end_i - start + 1);
        
        int framesPerSecond = context.sampleRate / context.len;
        context.bandHistory[context.oldestHistoryIndex] = energy;
        
        context.oldestHistoryIndex = (context.oldestHistoryIndex + 1) % framesPerSecond; // avoid shift
        
        float average = 0.0f;
        for (int i = 0; i < framesPerSecond; i++) {
            average += context.bandHistory[i];
        }
        average /= framesPerSecond;
        
        float variance = 0.0f;
        for (int i = 0; i < framesPerSecond; i++) {
            float diff = context.bandHistory[i] - average;
            variance += diff * diff;
        }
        variance /= framesPerSecond;
        
        float threshold = -0.0025714f * variance + 1.5142857f; // https://github.com/ddf/Minim/blob/main/src/main/java/ddf/minim/analysis/BeatDetect.java#L584 copilot gave values
        
        float time = get_time() * 1e3f; // technically should use output time but i'm so done :sob:
        float timeBetweenBeats = 100; // ms (600 bpm)
        
        if (energy - 0.05 > threshold * average && energy > 0.01f && (time - context.lastBeat) > timeBetweenBeats) {
            std::lock_guard<std::mutex> lock(mutex);
            context.aboveThreshold = energy - (threshold * average);
            context.lastBeat = time;
        } else {
            std::lock_guard<std::mutex> lock(mutex);
            context.aboveThreshold = 0.0f;
        }
        
        memcpy(context.buffer, context.buffer+context.len, (context.offset - context.len) * sizeof(float));
        context.offset = context.offset -= context.len;
    }
    
    return noErr;
}

class AudioInput {
private:
    AudioObjectID aggregateDeviceId = 0;
    AudioObjectID tapId = kAudioObjectUnknown;
    AudioDeviceIOProcID deviceProcId = 0;
public:
    FrameContext fftInput;
    AudioInput() {
        requestPermission();
        CATapDescription* description = [[CATapDescription alloc] initMonoGlobalTapButExcludeProcesses:@[]];

        printf("create tap %d\n", AudioHardwareCreateProcessTap(description, &tapId));
        uint32 propertySize = 0;
        AudioObjectPropertyAddress defaultDeviceAddress{.mSelector = kAudioHardwarePropertyDefaultSystemOutputDevice };
        AudioDeviceID defaultDevice = 0;
        uint32 size = sizeof(AudioObjectID);
        AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject), &defaultDeviceAddress, 0, nil, &size, &defaultDevice);
        AudioObjectPropertyAddress deviceUIDAddress{.mSelector = kAudioDevicePropertyDeviceUID };
        CFStringRef uuid;
        size = sizeof(CFStringRef);
        AudioObjectGetPropertyData(defaultDevice, &deviceUIDAddress, 0, nil, &size, &uuid);
        
        NSDictionary* dict = @{
            @kAudioAggregateDeviceNameKey: @"value1",
            @kAudioAggregateDeviceUIDKey: [[NSUUID UUID] UUIDString],
            @kAudioAggregateDeviceMainSubDeviceKey: (NSString*) uuid,
            @kAudioAggregateDeviceIsPrivateKey: @true,
            @kAudioAggregateDeviceTapAutoStartKey: @true,
            @kAudioAggregateDeviceIsStackedKey: @false,
            @kAudioAggregateDeviceSubDeviceListKey: @[
                @{
                    @kAudioSubDeviceUIDKey: (NSString*) uuid
                },
            ],
            @kAudioAggregateDeviceTapListKey: @[
                @{
                    @kAudioSubTapDriftCompensationKey: @true,
                    @kAudioSubTapUIDKey: (NSString*) [[description UUID] UUIDString]
                },
            ],
        };
        
        AudioObjectPropertyAddress formatAddress{.mSelector = kAudioTapPropertyFormat };
        AudioStreamBasicDescription streamDescription = AudioStreamBasicDescription();
        size = sizeof(streamDescription);
        AudioObjectGetPropertyData(tapId, &formatAddress, 0, nil, &size, &streamDescription);
        
        assert(streamDescription.mFormatID == kAudioFormatLinearPCM);
        
        int frame_size = 4096 / 4;
        int fps = 60;

        assert(frame_size / streamDescription.mSampleRate >= 1.0f / fps);
        
        int framesPerSecond = streamDescription.mSampleRate / frame_size;
        
        fftInput = {
            frame_size, 0, 0, (float) streamDescription.mSampleRate, 0, 0, precompute_fft_factors(frame_size), (float*) malloc(frame_size * sizeof(float) * 2), (float*) malloc(frame_size * sizeof(float)), (float*) malloc(framesPerSecond * sizeof(float)) };
        
        printf("create aggregate %d\n", AudioHardwareCreateAggregateDevice((CFDictionaryRef) dict, &aggregateDeviceId));
        printf("create callback %d\n", AudioDeviceCreateIOProcID(aggregateDeviceId, callback, &fftInput, &deviceProcId));
        printf("start device %d\n", AudioDeviceStart(aggregateDeviceId, deviceProcId));
    }
    
    ~AudioInput() {
        printf("deleted audio input\n");
        free(fftInput.buffer);
        free(fftInput.imags);
        free(fftInput.bandHistory);
        AudioDeviceStop(aggregateDeviceId, deviceProcId);
        AudioDeviceDestroyIOProcID(aggregateDeviceId, deviceProcId);
        AudioHardwareDestroyAggregateDevice(aggregateDeviceId);
        AudioHardwareDestroyProcessTap(tapId);
    }
};

Renderer* renderer = nil;
AudioInput* audioInput = nullptr;

typedef void (*func) (id, SEL);
IMP (present) = NULL;
void present_hook(id self, SEL sel) {
    renderer->updateFramebuffer(self);
    renderer->draw(self);
    ((func) present)(self, sel);
    return;
}

void hook(char* className, char* methodName, IMP* orig, IMP replacement) {
    Class klass = objc_getClass(className);
    Method method = class_getInstanceMethod(klass, sel_registerName(methodName));
    assert(method != nil);
    *orig = method_setImplementation(method, replacement);
}

__attribute__((constructor)) void init_hooks() {
    audioInput = new AudioInput();
    renderer = new Renderer(&audioInput->fftInput);
    hook("CAMetalDrawable", "present", &present, (IMP) present_hook);
}

__attribute__((destructor)) void destroy() {
    delete renderer;
    delete audioInput;
}


