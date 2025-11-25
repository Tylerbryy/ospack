; Python chunk targets for semantic code chunking

; Functions (sync and async)
(function_definition) @chunk.target
(async_function_definition) @chunk.target

; Classes
(class_definition) @chunk.target

; Decorated definitions (captures @decorator with the target)
(decorated_definition) @chunk.target
