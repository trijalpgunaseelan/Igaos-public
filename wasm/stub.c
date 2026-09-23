/* Single-threaded WASM build: thread_local destructors are registered through
   __cxa_thread_atexit, which WASI's threadless libc does not provide. The
   module runs one solve and exits, so nothing outlives the call; recording the
   destructor would change no observable behaviour. */
int __cxa_thread_atexit(void (*dtor)(void *), void *obj, void *dso) {
    (void)dtor; (void)obj; (void)dso;
    return 0;
}
