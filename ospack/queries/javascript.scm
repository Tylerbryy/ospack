; JavaScript/TypeScript chunk targets for semantic code chunking

; Function declarations
(function_declaration) @chunk.target

; Class declarations
(class_declaration) @chunk.target

; Method definitions inside classes
(method_definition) @chunk.target

; Arrow functions (when assigned to variables)
(arrow_function) @chunk.target

; Variable declarations with function values (const foo = () => ...)
(lexical_declaration) @chunk.target
