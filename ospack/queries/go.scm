; Go chunk targets for semantic code chunking

; Function declarations
(function_declaration) @chunk.target

; Method declarations
(method_declaration) @chunk.target

; Function literals (closures)
(func_literal) @chunk.target

; Type declarations (structs, interfaces)
(type_declaration) @chunk.target
