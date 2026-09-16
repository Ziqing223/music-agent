// Music Agent — local DeepSeek credential storage (P20 Slice B.1).
//
// The API key is stored only as a macOS generic-password Keychain item and
// is returned in memory solely for injection into the owned WebShell child.
// No operation logs or formats the credential value.
import Foundation
import Security

protocol DeepSeekCredentialReading {
    func read() throws -> String?
}

protocol DeepSeekCredentialStoring: DeepSeekCredentialReading {
    func save(_ credential: String) throws
}

enum DeepSeekCredentialSource: Equatable {
    case environment
    case keychain
}

struct ResolvedDeepSeekCredential: Equatable {
    let value: String
    let source: DeepSeekCredentialSource
}

enum DeepSeekCredentialResolution: Equatable {
    case available(ResolvedDeepSeekCredential)
    case missing
}

enum DeepSeekCredentialResolver {
    static func resolve(environment: [String: String],
                        store: DeepSeekCredentialReading) throws -> DeepSeekCredentialResolution {
        if let value = environment["DEEPSEEK_API_KEY"], !value.isEmpty {
            return .available(ResolvedDeepSeekCredential(value: value, source: .environment))
        }
        if let value = try store.read(), !value.isEmpty {
            return .available(ResolvedDeepSeekCredential(value: value, source: .keychain))
        }
        return .missing
    }
}

enum DeepSeekCredentialInputError: LocalizedError, Equatable {
    case empty
    case malformed

    var errorDescription: String? {
        switch self {
        case .empty:
            return "API Key 不能为空。"
        case .malformed:
            return "API Key 不能包含空白字符或引号。"
        }
    }
}

enum DeepSeekCredentialInput {
    static func normalize(_ rawValue: String) throws -> String {
        let value = rawValue.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !value.isEmpty else { throw DeepSeekCredentialInputError.empty }
        guard !value.contains(where: { $0.isWhitespace || $0 == "\"" || $0 == "'" }) else {
            throw DeepSeekCredentialInputError.malformed
        }
        return value
    }
}

enum KeychainCredentialStoreError: LocalizedError, Equatable {
    case unexpectedData
    case operationFailed(OSStatus)

    var errorDescription: String? {
        switch self {
        case .unexpectedData:
            return "Keychain 中的凭据格式无效。"
        case .operationFailed(let status):
            let detail = SecCopyErrorMessageString(status, nil) as String? ?? "OSStatus \(status)"
            return "Keychain 操作失败：\(detail)"
        }
    }
}

final class KeychainCredentialStore: DeepSeekCredentialStoring {
    static let service = "Music Agent"
    static let account = "DEEPSEEK_API_KEY"

    private let serviceName: String
    private let accountName: String

    init(service: String = KeychainCredentialStore.service,
         account: String = KeychainCredentialStore.account) {
        serviceName = service
        accountName = account
    }

    func read() throws -> String? {
        var query = baseQuery
        query[kSecReturnData as String] = true
        query[kSecMatchLimit as String] = kSecMatchLimitOne

        var result: CFTypeRef?
        let status = SecItemCopyMatching(query as CFDictionary, &result)
        if status == errSecItemNotFound { return nil }
        guard status == errSecSuccess else {
            throw KeychainCredentialStoreError.operationFailed(status)
        }
        guard let data = result as? Data,
              let value = String(data: data, encoding: .utf8) else {
            throw KeychainCredentialStoreError.unexpectedData
        }
        return value
    }

    func save(_ credential: String) throws {
        let data = Data(credential.utf8)
        let attributes = [kSecValueData as String: data]
        let updateStatus = SecItemUpdate(baseQuery as CFDictionary,
                                         attributes as CFDictionary)
        if updateStatus == errSecSuccess { return }
        guard updateStatus == errSecItemNotFound else {
            throw KeychainCredentialStoreError.operationFailed(updateStatus)
        }

        var item = baseQuery
        item[kSecValueData as String] = data
        item[kSecAttrSynchronizable as String] = false
        let addStatus = SecItemAdd(item as CFDictionary, nil)
        guard addStatus == errSecSuccess else {
            throw KeychainCredentialStoreError.operationFailed(addStatus)
        }
    }

    /// Test/support cleanup only; production UX intentionally never deletes
    /// a credential in response to provider authorization failures.
    func delete() throws {
        let status = SecItemDelete(baseQuery as CFDictionary)
        guard status == errSecSuccess || status == errSecItemNotFound else {
            throw KeychainCredentialStoreError.operationFailed(status)
        }
    }

    private var baseQuery: [String: Any] {
        [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: serviceName,
            kSecAttrAccount as String: accountName,
        ]
    }
}
