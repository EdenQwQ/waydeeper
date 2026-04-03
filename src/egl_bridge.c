// EGL bridge for Wayland - creates EGL context/surface from raw Wayland pointers.
// This avoids fighting with Rust type systems for EGL native types.

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <wayland-client.h>
#include <wayland-egl.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    EGLDisplay display;
    EGLContext context;
    EGLConfig config;
} EglCtx;

// Initialize EGL from a Wayland display pointer.
// Returns 0 on success, -1 on failure.
int egl_init_from_wl_display(void *wl_display_ptr, EglCtx *out) {
    EGLDisplay egl_display = EGL_NO_DISPLAY;

    // Use eglGetPlatformDisplay for Wayland (EGL 1.5 or EGL_EXT_platform_base).
    // This is the standard Wayland EGL init — raw eglGetDisplay doesn't route
    // correctly through libglvnd on some systems.
    typedef EGLDisplay (*PFNEGLGETPLATFORMDISPLAYEXTPROC)(EGLenum, void *, const EGLAttrib *);
    PFNEGLGETPLATFORMDISPLAYEXTPROC get_platform_display =
        (PFNEGLGETPLATFORMDISPLAYEXTPROC)eglGetProcAddress("eglGetPlatformDisplayEXT");
    if (get_platform_display) {
        // EGL_PLATFORM_WAYLAND_KHR = 0x31D8
        egl_display = get_platform_display(0x31D8, wl_display_ptr, NULL);
    }
    if (egl_display == EGL_NO_DISPLAY) {
        // Fallback: try the raw pointer through eglGetDisplay
        egl_display = eglGetDisplay((EGLNativeDisplayType)wl_display_ptr);
    }
    if (egl_display == EGL_NO_DISPLAY) {
        // Last resort: EGL_DEFAULT_DISPLAY
        egl_display = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    }
    if (egl_display == EGL_NO_DISPLAY) {
        fprintf(stderr, "egl_bridge: eglGetDisplay failed\n");
        return -1;
    }

    EGLint major, minor;
    if (!eglInitialize(egl_display, &major, &minor)) {
        fprintf(stderr, "egl_bridge: eglInitialize failed (error 0x%x)\n", eglGetError());
        return -1;
    }

    fprintf(stderr, "egl_bridge: EGL %d.%d initialized\n", major, minor);

    EGLint config_attribs[] = {
        EGL_SURFACE_TYPE, EGL_WINDOW_BIT,
        EGL_RENDERABLE_TYPE, EGL_OPENGL_ES2_BIT,
        EGL_RED_SIZE, 8,
        EGL_GREEN_SIZE, 8,
        EGL_BLUE_SIZE, 8,
        EGL_ALPHA_SIZE, 8,
        EGL_NONE
    };

    EGLConfig config;
    EGLint num_configs;
    if (!eglChooseConfig(egl_display, config_attribs, &config, 1, &num_configs) || num_configs == 0) {
        fprintf(stderr, "egl_bridge: eglChooseConfig failed\n");
        return -1;
    }

    EGLint ctx_attribs[] = {
        EGL_CONTEXT_MAJOR_VERSION, 3,
        EGL_CONTEXT_MINOR_VERSION, 0,
        EGL_NONE
    };

    EGLContext egl_context = eglCreateContext(egl_display, config, EGL_NO_CONTEXT, ctx_attribs);
    if (egl_context == EGL_NO_CONTEXT) {
        // Try ES 2.0 as fallback
        ctx_attribs[1] = 2;
        ctx_attribs[3] = 0;
        egl_context = eglCreateContext(egl_display, config, EGL_NO_CONTEXT, ctx_attribs);
        if (egl_context == EGL_NO_CONTEXT) {
            fprintf(stderr, "egl_bridge: eglCreateContext failed (error 0x%x)\n", eglGetError());
            return -1;
        }
        fprintf(stderr, "egl_bridge: Using OpenGL ES 2.0 context\n");
    } else {
        fprintf(stderr, "egl_bridge: Using OpenGL ES 3.0 context\n");
    }

    out->display = egl_display;
    out->context = egl_context;
    out->config = config;
    return 0;
}

// Create an EGL window surface from a wl_surface pointer.
// Returns the EGLSurface, or EGL_NO_SURFACE on failure.
void* egl_create_surface(EglCtx *ctx, void *wl_surface_ptr, int width, int height) {
    // Create a wl_egl_window from the wl_surface
    struct wl_egl_window *egl_window = wl_egl_window_create(
        (struct wl_surface *)wl_surface_ptr, width, height);
    if (!egl_window) {
        fprintf(stderr, "egl_bridge: wl_egl_window_create failed\n");
        return NULL;
    }

    EGLSurface egl_surface = eglCreateWindowSurface(
        ctx->display, ctx->config, (EGLNativeWindowType)egl_window, NULL);
    if (egl_surface == EGL_NO_SURFACE) {
        fprintf(stderr, "egl_bridge: eglCreateWindowSurface failed (error 0x%x)\n", eglGetError());
        wl_egl_window_destroy(egl_window);
        return NULL;
    }

    // Make context current
    if (!eglMakeCurrent(ctx->display, egl_surface, egl_surface, ctx->context)) {
        fprintf(stderr, "egl_bridge: eglMakeCurrent failed (error 0x%x)\n", eglGetError());
        eglDestroySurface(ctx->display, egl_surface);
        wl_egl_window_destroy(egl_window);
        return NULL;
    }

    fprintf(stderr, "egl_bridge: EGL surface created %dx%d\n", width, height);
    return (void*)egl_surface;
}

// Resize the EGL window surface.
void egl_resize_surface(EglCtx *ctx, void *egl_surface_ptr, int width, int height) {
    // We need the wl_egl_window. Since we can't easily pass it back,
    // we'll need to track it separately. For now, this is a no-op
    // and the caller should recreate the surface on resize.
    (void)ctx;
    (void)egl_surface_ptr;
    (void)width;
    (void)height;
}

// Swap buffers.
int egl_swap_buffers(EglCtx *ctx, void *egl_surface_ptr) {
    return eglSwapBuffers(ctx->display, (EGLSurface)egl_surface_ptr);
}

// Get EGL proc address.
void* egl_get_proc_address(const char *name) {
    return (void*)eglGetProcAddress(name);
}

// Destroy EGL surface.
void egl_destroy_surface(EglCtx *ctx, void *egl_surface_ptr) {
    eglMakeCurrent(ctx->display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
    eglDestroySurface(ctx->display, (EGLSurface)egl_surface_ptr);
}

// Destroy EGL context.
void egl_destroy_ctx(EglCtx *ctx) {
    eglMakeCurrent(ctx->display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
    eglDestroyContext(ctx->display, ctx->context);
    eglTerminate(ctx->display);
}
